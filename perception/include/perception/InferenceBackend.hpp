#pragma once

#include "perception/Detection.hpp"
#include "perception/Frame.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// 단일 이미지 추론 한 번의 단계별 소요 시간 (ms)
// - preprocess : 입력 이미지를 모델 입력 형태로 변환 (letterbox, blob 생성 등)
// - inference  : 추론 엔진 실행 (입력 바인딩 + forward)
// - postprocess: 출력 텐서 디코드, 좌표 복원, 클래스 필터, 단일 이미지 NMS
//
// 세부 타이머 (preprocess 안의 몫). 그 경로에 없는 단계는 비어 있음 → raw_frame_log 빈 칸
// - resize / pad / blob : CPU letterbox 의 cv::resize, copyMakeBorder, blobFromImage
// - pinnedCopy          : blob → pinned 버퍼 memcpy (TensorRT CPU 경로만)
// - gpuPreprocess       : 프레임 업로드 + 커널 + sync 벽시계 (TensorRT --gpu-preprocess 경로만. 이때 preprocess 와 같은 값)
//                         v2(--camera-zero-copy)는 업로드가 없어 커널 + sync 만임
struct InferenceTiming {
    double preprocessMilliseconds = 0.0;
    double inferenceMilliseconds = 0.0;
    double postprocessMilliseconds = 0.0;
    std::optional<double> resizeMilliseconds;
    std::optional<double> padMilliseconds;
    std::optional<double> blobMilliseconds;
    std::optional<double> pinnedCopyMilliseconds;
    std::optional<double> gpuPreprocessMilliseconds;
};

/*
 * 전처리 경로 (--gpu-preprocess). 기본은 CPU 경로 (golden 보존)
 *   gpu   : TensorRT 전용. GPU 커널 한 번으로 letterbox·RGB·/255 를 해서 TensorRT 입력 버퍼에 바로 씀
 *           (CPU resize·pad·blob, pinned memcpy, blob H2D 가 빠지고 프레임 업로드가 들어감)
 *   check : gpu 와 같이 씀. 프레임마다 GPU 입력 텐서를 CPU 경로(letterbox + blobFromImage)·CPU 기준 함수와 비트 단위로 비교해
 *           다른 값 수를 셈. 끝날 때 요약을 표준 에러로 냄 (확인용이라 느림. 측정에는 안 씀)
 */
struct PreprocessOptions {
    bool gpu = false;
    bool check = false;
};

/*
 * 추론 백엔드 계약: 이미지 한 장 → 입력 이미지 좌표계의 Detection 목록
 *
 * 전처리 → 추론 → 출력 디코드/후처리(단일 이미지 NMS 포함)까지
 * 구현체가 전부 처리한다. 상위 계층은 blob, NCHW, 출력 텐서 형태,
 * 추론 엔진 종류를 알지 못한다.
 *
 * 구현체는 전달받은 이미지가 전체 프레임인지 crop인지 알지 못한다.
 * Full + Crop 같은 호출 정책은 YoloDetector가 담당한다.
 */
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // timing이 nullptr가 아니면 단계별 소요 시간을 채운다
    virtual std::vector<Detection> infer(const cv::Mat& image, InferenceTiming* timing = nullptr) = 0;

    // GPU 전처리 경로면 true. 이때 YoloDetector 는 Crop 을 clone 하지 않고 ROI 뷰로 넘김 (업로드가 행 간격을 받음)
    virtual bool usesGpuPreprocess() const { return false; }

    // GPU 전처리 v2 (--camera-zero-copy): 카메라 YUYV 버퍼의 roi 영역을 GPU 가 직접 읽어 입력을 만듦 (업로드 없음)
    // image 는 같은 영역의 BGR (cvtColor 결과). 좌표 복원 크기와 --gpu-preprocess-check 비교에만 씀
    // roi 는 YUYV 버퍼 안의 좌표 (Full = 화면 전체, Crop = crop 창). GPU 전처리 백엔드(TensorRT)만 지원함
    virtual std::vector<Detection> inferYuyv(const cv::Mat& image, const YuyvBuffer& yuyv, const cv::Rect& roi, InferenceTiming* timing = nullptr) {
        (void)image;
        (void)yuyv;
        (void)roi;
        (void)timing;
        throw std::logic_error("이 백엔드는 카메라 YUYV 직접 입력(--camera-zero-copy)을 지원하지 않음");
    }
};

/*
 * tensorrt_int8 실험용 설정. 정식 엔진(variant 비움)은 이 값들이 전부 기본이라 결과가 안 바뀜
 *   - variant     : 비어 있으면 <모델>.int8.engine / .calib. 아니면 <모델>.int8-<variant>.engine / .calib 로 파일을 분리함
 *                   → 실험 엔진이 정식 엔진·golden 을 덮어쓰지 않음
 *   - calibrator  : "entropy" (IInt8EntropyCalibrator2, 기본) | "minmax" (IInt8MinMaxCalibrator)
 *   - shuffleSeed : 0 이상이면 calibration 이미지 순서를 이 시드로 섞음 (-1 = 목록 순서 그대로)
 *   - fp32Head    : true 면 검출 헤드(/model.22/) 층을 전부 FP32 로 강제함 (kOBEY_PRECISION_CONSTRAINTS)
 */
struct Int8Tuning {
    std::string variant;
    std::string calibrator = "entropy";
    int shuffleSeed = -1;
    bool fp32Head = false;
};

/*
 * backendName으로 추론 백엔드를 생성한다.
 *   "opencv_dnn" : OpenCV DNN (CPU)
 * 알 수 없는 이름이면 std::invalid_argument,
 * 모델 로드 실패는 구현체의 예외(cv::Exception 등)를 그대로 전달한다.
 *
 * inputSize 는 모델의 입력 크기(가로 x 세로). 기본 640x640.
 *   - OpenCV DNN : 이 크기로 letterbox 와 blob 을 만듦 (ONNX 입력과 맞아야 함)
 *   - TensorRT   : 엔진에서 읽은 입력 크기와 다르면 예외 (엔진 캐시 오류 방지)
 *
 * calibrationList 는 tensorrt_int8 전용. Calibration 이미지 경로 목록 파일(한 줄에 하나).
 *   INT8 엔진을 처음 만들 때만 필요하고, 엔진이나 calibration 캐시가 이미 있으면 비워도 됨.
 *   다른 백엔드는 무시함.
 * int8Tuning 도 tensorrt_int8 전용 (위 Int8Tuning). 기본값이면 정식 엔진과 같음.
 * preprocess 는 전처리 경로 (위 PreprocessOptions). gpu 는 tensorrt_* 만 받고, 다른 백엔드면 std::invalid_argument.
 */
std::unique_ptr<InferenceBackend> createInferenceBackend(
    const std::string& backendName,
    const std::string& modelPath,
    float confidenceThreshold,
    float nmsThreshold,
    const cv::Size& inputSize = cv::Size(640, 640),
    const std::string& calibrationList = "",
    const Int8Tuning& int8Tuning = Int8Tuning(),
    const PreprocessOptions& preprocess = PreprocessOptions()
);