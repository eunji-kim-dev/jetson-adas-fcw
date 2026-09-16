#include "TensorRTBackend.hpp"

#include "YoloCommon.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/dnn.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ------------------------------------------------------------
// 작은 도우미들
// ------------------------------------------------------------

// CUDA 호출 결과를 검사하고 실패면 예외로 바꿈
void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA 실패 (") + what + "): " + cudaGetErrorString(status));
    }
}

// TensorRT 로그를 표준 에러로 흘림. INFO 이하는 너무 많아서 버림
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity > Severity::kWARNING) return;
        const char* tag = severity == Severity::kWARNING ? "WARN" : "ERROR";
        std::cerr << "[TensorRT " << tag << "] " << message << '\n';
    }
};

// Dims 의 원소 개수
std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t total = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("동적 차원은 지원하지 않음 (static shape ONNX 필요)");
        total *= static_cast<std::size_t>(dims.d[i]);
    }
    return total;
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string out = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i) out += ", ";
        out += std::to_string(dims.d[i]);
    }
    return out + "]";
}

// <모델>.onnx → <모델>.<정밀도>.engine
std::string engineFileFor(const std::string& modelPath, const std::string& precision) {
    std::filesystem::path p(modelPath);
    p.replace_extension("." + precision + ".engine");
    return p.string();
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// 엔진이 없거나 ONNX 가 엔진보다 새로우면 다시 빌드해야 함
bool engineIsStale(const std::string& modelPath, const std::string& enginePath) {
    namespace fs = std::filesystem;
    if (!fs::exists(enginePath)) return true;
    if (!fs::exists(modelPath)) return false;
    return fs::last_write_time(modelPath) > fs::last_write_time(enginePath);
}

// ------------------------------------------------------------
// ONNX → 직렬화된 엔진
// ------------------------------------------------------------

std::vector<char> buildEngine(const std::string& modelPath, const std::string& precision, nvinfer1::ILogger& logger) {
    using namespace nvinfer1;

    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) throw std::runtime_error("TensorRT builder 생성 실패");

    // TensorRT 10 은 항상 explicit batch 라 플래그 0 이면 됨
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network) throw std::runtime_error("TensorRT network 생성 실패");

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) throw std::runtime_error("ONNX parser 생성 실패");

    if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::string reason;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            reason += std::string("\n  ") + parser->getError(i)->desc();
        }
        throw std::runtime_error("ONNX 파싱 실패: " + modelPath + reason);
    }

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) throw std::runtime_error("TensorRT builder config 생성 실패");

    // 빌드 중 커널 탐색용 작업 메모리. yolov8n 은 256MB 면 충분함
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 256ULL << 20);

    if (precision == "fp16") {
        // kFP16 은 TensorRT 10.12 부터 deprecated (강타입 네트워크로 대체 권고) 지만 아직 동작함.
        // 같은 FP32 ONNX 로 FP32/FP16/INT8 을 비교하는 게 목적이라 약타입 방식이 맞음.
        // 강타입으로 가려면 FP16 으로 export 한 ONNX 가 따로 필요해서 비교 조건이 달라짐.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->setFlag(BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    } else if (precision == "int8") {
        throw std::runtime_error("int8 은 아직 미구현 (calibrator 필요)");
    } else if (precision != "fp32") {
        throw std::runtime_error("모르는 정밀도: " + precision + " (fp32 | fp16)");
    }

    std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("TensorRT 엔진 빌드 실패");

    const char* begin = static_cast<const char*>(serialized->data());
    return std::vector<char>(begin, begin + serialized->size());
}

} // namespace

// ------------------------------------------------------------
// Impl: TensorRT / CUDA 자원을 전부 여기 둠
// ------------------------------------------------------------
// 멤버 선언 순서가 곧 파괴 역순임.
// context → engine → runtime 순으로 지워져야 하므로 runtime 을 먼저 선언함.

struct TensorRTBackend::Impl {
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream = nullptr;

    std::string inputName;
    std::string outputName;
    nvinfer1::Dims inputDims{};
    nvinfer1::Dims outputDims{};
    std::size_t inputBytes = 0;
    std::size_t outputBytes = 0;
    int inputSize = 0;   // 정사각형 입력 한 변 (640)

    // 호스트는 pinned, 디바이스는 cudaMalloc
    float* hostInput = nullptr;
    float* hostOutput = nullptr;
    void* deviceInput = nullptr;
    void* deviceOutput = nullptr;

    ~Impl() {
        // 실행 중인 작업이 있으면 끝난 뒤에 지움
        if (stream) cudaStreamSynchronize(stream);
        if (deviceInput) cudaFree(deviceInput);
        if (deviceOutput) cudaFree(deviceOutput);
        if (hostInput) cudaFreeHost(hostInput);
        if (hostOutput) cudaFreeHost(hostOutput);
        if (stream) cudaStreamDestroy(stream);
        // context / engine / runtime 는 unique_ptr 가 선언 역순으로 지움
    }
};

// ------------------------------------------------------------
// 생성자: 엔진 준비 → I/O 텐서 조사 → 버퍼 할당 → 주소 바인딩
// ------------------------------------------------------------

TensorRTBackend::TensorRTBackend(const std::string& modelPath, const std::string& precision, float confidenceThreshold, float nmsThreshold)
    : impl_(std::make_unique<Impl>()), confidenceThreshold_(confidenceThreshold), nmsThreshold_(nmsThreshold) {
    using namespace nvinfer1;

    enginePath_ = engineFileFor(modelPath, precision);

    // 1) 엔진 읽기. 오래됐거나 없거나 못 읽으면 빌드
    std::vector<char> blob;
    if (!engineIsStale(modelPath, enginePath_)) {
        blob = readFile(enginePath_);
    }

    impl_->runtime.reset(createInferRuntime(impl_->logger));
    if (!impl_->runtime) throw std::runtime_error("TensorRT runtime 생성 실패");

    if (!blob.empty()) {
        impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!impl_->engine) {
            std::cerr << "[TensorRT WARN] 엔진 파일을 못 읽음 (다른 GPU/버전에서 만든 것일 수 있음) — 다시 빌드함: " << enginePath_ << '\n';
        }
    }

    if (!impl_->engine) {
        std::cerr << "[TensorRT] 엔진 빌드 시작 (" << precision << "): " << modelPath << " — Jetson 에서 수 분 걸림\n";
        const auto buildStart = std::chrono::steady_clock::now();
        blob = buildEngine(modelPath, precision, impl_->logger);
        const double buildSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - buildStart).count();

        std::ofstream out(enginePath_, std::ios::binary);
        if (!out) throw std::runtime_error("엔진 파일 저장 실패: " + enginePath_);
        out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
        out.close();
        engineWasBuilt_ = true;
        std::cerr << "[TensorRT] 엔진 빌드 완료 " << buildSeconds << "s → " << enginePath_ << '\n';

        impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!impl_->engine) throw std::runtime_error("방금 빌드한 엔진을 못 읽음: " + enginePath_);
    }

    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) throw std::runtime_error("TensorRT execution context 생성 실패");

    // 2) I/O 텐서 조사. yolov8n 은 입력 1개(images), 출력 1개(output0)
    const int tensorCount = impl_->engine->getNbIOTensors();
    for (int i = 0; i < tensorCount; ++i) {
        const char* name = impl_->engine->getIOTensorName(i);
        const TensorIOMode mode = impl_->engine->getTensorIOMode(name);
        const Dims dims = impl_->engine->getTensorShape(name);
        const DataType type = impl_->engine->getTensorDataType(name);

        if (type != DataType::kFLOAT) {
            throw std::runtime_error(std::string("I/O 텐서가 float32 가 아님: ") + name);
        }

        if (mode == TensorIOMode::kINPUT) {
            if (!impl_->inputName.empty()) throw std::runtime_error("입력 텐서가 두 개 이상임");
            impl_->inputName = name;
            impl_->inputDims = dims;
        } else if (mode == TensorIOMode::kOUTPUT) {
            if (!impl_->outputName.empty()) throw std::runtime_error("출력 텐서가 두 개 이상임");
            impl_->outputName = name;
            impl_->outputDims = dims;
        }
    }
    if (impl_->inputName.empty() || impl_->outputName.empty()) {
        throw std::runtime_error("입력 또는 출력 텐서를 못 찾음");
    }

    // 입력은 [1, 3, H, W] 정사각형이어야 letterbox 를 그대로 쓸 수 있음
    const Dims& in = impl_->inputDims;
    if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != in.d[3]) {
        throw std::runtime_error("지원하지 않는 입력 형태 " + dimsToString(in) + " (기대: [1, 3, N, N])");
    }
    impl_->inputSize = static_cast<int>(in.d[2]);

    // 출력은 [1, 84, 8400] 처럼 3차원
    if (impl_->outputDims.nbDims != 3 || impl_->outputDims.d[0] != 1) {
        throw std::runtime_error("지원하지 않는 출력 형태 " + dimsToString(impl_->outputDims) + " (기대: [1, C, N])");
    }

    impl_->inputBytes = volume(impl_->inputDims) * sizeof(float);
    impl_->outputBytes = volume(impl_->outputDims) * sizeof(float);

    // 3) 버퍼 할당. 호스트는 pinned 로 잡아야 비동기 복사가 실제로 비동기가 됨
    checkCuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    checkCuda(cudaMalloc(&impl_->deviceInput, impl_->inputBytes), "cudaMalloc(input)");
    checkCuda(cudaMalloc(&impl_->deviceOutput, impl_->outputBytes), "cudaMalloc(output)");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&impl_->hostInput), impl_->inputBytes), "cudaMallocHost(input)");
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&impl_->hostOutput), impl_->outputBytes), "cudaMallocHost(output)");

    // 4) 주소 바인딩은 버퍼가 고정이라 한 번만 하면 됨
    if (!impl_->context->setTensorAddress(impl_->inputName.c_str(), impl_->deviceInput)) {
        throw std::runtime_error("입력 텐서 주소 바인딩 실패: " + impl_->inputName);
    }
    if (!impl_->context->setTensorAddress(impl_->outputName.c_str(), impl_->deviceOutput)) {
        throw std::runtime_error("출력 텐서 주소 바인딩 실패: " + impl_->outputName);
    }

    std::cerr << "[TensorRT] 준비 완료 " << precision
              << " 입력 " << impl_->inputName << dimsToString(impl_->inputDims)
              << " 출력 " << impl_->outputName << dimsToString(impl_->outputDims) << '\n';
}

TensorRTBackend::~TensorRTBackend() = default;

// ------------------------------------------------------------
// 추론
// ------------------------------------------------------------
// 단계 구분은 OpenCVDNNBackend 와 맞춤
//   preprocess  : letterbox + blob (CPU)
//   inference   : H2D 복사 + enqueue + D2H 복사 + 동기화
//   postprocess : 디코드 + NMS (CPU)

std::vector<Detection> TensorRTBackend::infer(const cv::Mat& image, InferenceTiming* timing) {
    Impl& t = *impl_;

    // 전처리
    const auto preprocessStart = std::chrono::steady_clock::now();
    const LetterboxResult prepared = letterbox(image, t.inputSize);
    cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, cv::Size(t.inputSize, t.inputSize), cv::Scalar(), true, false);

    if (!blob.isContinuous() || blob.total() * sizeof(float) != t.inputBytes) {
        throw std::runtime_error("blob 크기가 엔진 입력과 다름");
    }
    // 한 번 더 복사하는 이유: blob 은 일반 메모리라 pinned 버퍼로 옮겨야 비동기 H2D 가 됨
    // (Memory Path 단계에서 blob 을 pinned 에 직접 만들거나 GPU 전처리로 바꿀 자리)
    std::memcpy(t.hostInput, blob.ptr<float>(), t.inputBytes);
    const auto preprocessEnd = std::chrono::steady_clock::now();

    // 추론
    checkCuda(cudaMemcpyAsync(t.deviceInput, t.hostInput, t.inputBytes, cudaMemcpyHostToDevice, t.stream), "H2D");
    if (!t.context->enqueueV3(t.stream)) {
        throw std::runtime_error("TensorRT enqueueV3 실패");
    }
    checkCuda(cudaMemcpyAsync(t.hostOutput, t.deviceOutput, t.outputBytes, cudaMemcpyDeviceToHost, t.stream), "D2H");
    checkCuda(cudaStreamSynchronize(t.stream), "cudaStreamSynchronize");
    const auto inferenceEnd = std::chrono::steady_clock::now();

    // 후처리. 계산은 OpenCVDNNBackend 와 같은 YoloCommon 코드를 씀
    std::vector<Detection> detections = decodeYoloOutput(
        t.hostOutput,
        static_cast<int>(t.outputDims.d[1]),
        static_cast<int>(t.outputDims.d[2]),
        prepared, image.size(), confidenceThreshold_, nmsThreshold_);
    const auto postprocessEnd = std::chrono::steady_clock::now();

    if (timing != nullptr) {
        timing->preprocessMilliseconds = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        timing->inferenceMilliseconds = std::chrono::duration<double, std::milli>(inferenceEnd - preprocessEnd).count();
        timing->postprocessMilliseconds = std::chrono::duration<double, std::milli>(postprocessEnd - inferenceEnd).count();
    }

    return detections;
}