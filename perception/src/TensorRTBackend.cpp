#include "TensorRTBackend.hpp"

#include "YoloCommon.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
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

// <모델>.onnx → <모델>.int8.calib (calibration 캐시)
std::string calibrationCacheFileFor(const std::string& modelPath) {
    std::filesystem::path p(modelPath);
    p.replace_extension(".int8.calib");
    return p.string();
}

// calibration 목록 파일을 읽음. 한 줄에 이미지 경로 하나, 빈 줄과 # 줄은 무시함
std::vector<std::string> readCalibrationList(const std::string& listPath) {
    std::vector<std::string> paths;
    if (listPath.empty()) return paths;
    std::ifstream in(listPath);
    if (!in) throw std::runtime_error("calibration 목록 파일을 못 읽음: " + listPath);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        paths.push_back(line);
    }
    return paths;
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
// INT8 calibrator
// ------------------------------------------------------------
// TensorRT 가 엔진을 만들면서 이미지를 한 장씩 달라고 하면(getBatch) 넘겨줌.
// 전처리는 실행 때와 똑같이 letterbox + /255 + RGB 로 해야 스케일이 맞음.
// 결과(층별 스케일)는 캐시 파일에 저장해 두고, 있으면 이미지를 안 읽고 그걸 씀.
//
// IInt8EntropyCalibrator2 와 kINT8 플래그는 TensorRT 10.1 부터 deprecated (Q/DQ explicit quantization 권고)
// 지만 아직 동작함. 같은 FP32 ONNX 로 FP32/FP16/INT8 을 비교하는 게 목적이라 이 방식이 맞음.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8Calibrator(std::vector<std::string> imagePaths, std::string cachePath, const cv::Size& inputSize, std::size_t inputBytes)
        : imagePaths_(std::move(imagePaths)), cachePath_(std::move(cachePath)), inputSize_(inputSize), inputBytes_(inputBytes) {
        checkCuda(cudaMalloc(&deviceInput_, inputBytes_), "cudaMalloc(calibration input)");
    }

    ~Int8Calibrator() override {
        if (deviceInput_) cudaFree(deviceInput_);
    }

    Int8Calibrator(const Int8Calibrator&) = delete;
    Int8Calibrator& operator=(const Int8Calibrator&) = delete;

    // ONNX 배치 차원이 1 로 고정돼 있어 한 장씩 넘김
    int32_t getBatchSize() const noexcept override { return 1; }

    // 다음 이미지를 전처리해서 GPU 입력 버퍼에 올림. 더 없으면 false → calibration 끝
    bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override {
        (void)names;
        if (nbBindings < 1) return false;
        while (nextIndex_ < imagePaths_.size()) {
            const std::string& path = imagePaths_[nextIndex_++];
            cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "[TensorRT WARN] calibration 이미지를 못 읽음, 건너뜀: " << path << '\n';
                continue;
            }
            // 실행 때(TensorRTBackend::infer)와 같은 전처리
            const LetterboxResult prepared = letterbox(image, inputSize_);
            cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false);
            if (!blob.isContinuous() || blob.total() * sizeof(float) != inputBytes_) {
                std::cerr << "[TensorRT WARN] calibration blob 크기가 입력과 다름, 건너뜀: " << path << '\n';
                continue;
            }
            if (cudaMemcpy(deviceInput_, blob.ptr<float>(), inputBytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                std::cerr << "[TensorRT ERROR] calibration H2D 복사 실패\n";
                return false;
            }
            bindings[0] = deviceInput_;
            ++usedCount_;
            if (usedCount_ % 50 == 0) std::cerr << "[TensorRT] calibration " << usedCount_ << " / " << imagePaths_.size() << '\n';
            return true;
        }
        return false;
    }

    // 캐시가 있으면 그걸 돌려줌 → TensorRT 가 getBatch 를 안 부름
    const void* readCalibrationCache(std::size_t& length) noexcept override {
        cache_ = readFile(cachePath_);
        length = cache_.size();
        if (!cache_.empty()) std::cerr << "[TensorRT] calibration 캐시 사용: " << cachePath_ << '\n';
        return cache_.empty() ? nullptr : cache_.data();
    }

    // 새로 계산한 스케일을 캐시 파일로 저장
    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override {
        std::ofstream out(cachePath_, std::ios::binary);
        if (!out) {
            std::cerr << "[TensorRT WARN] calibration 캐시 저장 실패: " << cachePath_ << '\n';
            return;
        }
        out.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(length));
        std::cerr << "[TensorRT] calibration 캐시 저장 (" << usedCount_ << " 장): " << cachePath_ << '\n';
    }

    std::size_t usedCount() const { return usedCount_; }

private:
    std::vector<std::string> imagePaths_;
    std::string cachePath_;
    cv::Size inputSize_;
    std::size_t inputBytes_;
    std::size_t nextIndex_ = 0;
    std::size_t usedCount_ = 0;
    void* deviceInput_ = nullptr;
    std::vector<char> cache_;
};
#pragma GCC diagnostic pop

// ------------------------------------------------------------
// ONNX → 직렬화된 엔진
// ------------------------------------------------------------

std::vector<char> buildEngine(const std::string& modelPath, const std::string& precision, const std::string& calibrationList, nvinfer1::ILogger& logger) {
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

    // calibrator 는 buildSerializedNetwork 가 끝날 때까지 살아 있어야 해서 이 함수 범위에 둠
    std::unique_ptr<Int8Calibrator> calibrator;

    if (precision == "fp16") {
        // kFP16 은 TensorRT 10.12 부터 deprecated (강타입 네트워크로 대체 권고) 지만 아직 동작함.
        // 같은 FP32 ONNX 로 FP32/FP16/INT8 을 비교하는 게 목적이라 약타입 방식이 맞음.
        // 강타입으로 가려면 FP16 으로 export 한 ONNX 가 따로 필요해서 비교 조건이 달라짐.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->setFlag(BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    } else if (precision == "int8") {
        // 네트워크 입력 크기 [1, 3, H, W] 로 calibration 전처리 크기를 정함 (실행 때와 같은 letterbox)
        if (network->getNbInputs() < 1) throw std::runtime_error("네트워크 입력이 없음");
        const Dims in = network->getInput(0)->getDimensions();
        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] < 1 || in.d[3] < 1) {
            throw std::runtime_error("int8 calibration 은 [1, 3, H, W] 고정 입력만 지원함: " + dimsToString(in));
        }
        const cv::Size inputSize(static_cast<int>(in.d[3]), static_cast<int>(in.d[2]));
        const std::size_t inputBytes = volume(in) * sizeof(float);

        const std::string cachePath = calibrationCacheFileFor(modelPath);
        const std::vector<std::string> images = readCalibrationList(calibrationList);
        if (images.empty() && !std::filesystem::exists(cachePath)) {
            throw std::runtime_error(
                "int8 엔진을 만들려면 calibration 이미지 목록(--calib-list / --crop-calib-list)이나 캐시 파일이 필요함: " + cachePath);
        }
        std::cerr << "[TensorRT] int8 calibration: 이미지 " << images.size() << " 장, 캐시 " << cachePath
                  << (std::filesystem::exists(cachePath) ? " (있음, 이미지 대신 캐시 사용)" : " (없음, 새로 계산)") << '\n';

        calibrator = std::make_unique<Int8Calibrator>(images, cachePath, inputSize, inputBytes);

        // kINT8 / setInt8Calibrator 도 10.1 부터 deprecated 지만 동작함 (위 kFP16 과 같은 이유로 이 방식을 씀)
        // INT8 로 못 만드는 층은 FP16 으로 떨어지게 kFP16 도 같이 켬
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);
        config->setInt8Calibrator(calibrator.get());
#pragma GCC diagnostic pop
    } else if (precision != "fp32") {
        throw std::runtime_error("모르는 정밀도: " + precision + " (fp32 | fp16 | int8)");
    }

    std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) throw std::runtime_error("TensorRT 엔진 빌드 실패");

    if (calibrator && calibrator->usedCount() > 0) {
        std::cerr << "[TensorRT] int8 calibration 에 쓴 이미지: " << calibrator->usedCount() << " 장\n";
    }

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
    cv::Size inputSize;  // 모델 입력 크기(가로 x 세로). 640x640 또는 288x640 등

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

TensorRTBackend::TensorRTBackend(const std::string& modelPath, const std::string& precision, float confidenceThreshold, float nmsThreshold,
                                 const cv::Size& expectedInputSize, const std::string& calibrationList)
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
        blob = buildEngine(modelPath, precision, calibrationList, impl_->logger);
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

    // 입력은 [1, 3, H, W]. 정사각형이 아니어도 됨 (Crop 용 288x640)
    const Dims& in = impl_->inputDims;
    if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3) {
        throw std::runtime_error("지원하지 않는 입력 형태 " + dimsToString(in) + " (기대: [1, 3, H, W])");
    }
    impl_->inputSize = cv::Size(static_cast<int>(in.d[3]), static_cast<int>(in.d[2]));

    // 호출자가 기대한 크기와 엔진 크기가 다르면 멈춤
    // (예: 288x640 ONNX 를 줬는데 640x640 엔진 캐시를 읽어 버린 경우)
    if (impl_->inputSize != expectedInputSize) {
        throw std::runtime_error("엔진 입력 크기 " + std::to_string(impl_->inputSize.height) + "x" + std::to_string(impl_->inputSize.width)
            + " 가 기대한 크기 " + std::to_string(expectedInputSize.height) + "x" + std::to_string(expectedInputSize.width) + " 와 다름 (HxW): " + enginePath_);
    }

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
    cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, t.inputSize, cv::Scalar(), true, false);
    
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