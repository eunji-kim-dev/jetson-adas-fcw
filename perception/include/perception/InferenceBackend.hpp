#pragma once

#include "perception/Detection.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

// 단일 이미지 추론 한 번의 단계별 소요 시간 (ms)
// - preprocess : 입력 이미지를 모델 입력 형태로 변환 (letterbox, blob 생성 등)
// - inference  : 추론 엔진 실행 (입력 바인딩 + forward)
// - postprocess: 출력 텐서 디코드, 좌표 복원, 클래스 필터, 단일 이미지 NMS
struct InferenceTiming {
    double preprocessMilliseconds = 0.0;
    double inferenceMilliseconds = 0.0;
    double postprocessMilliseconds = 0.0;
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
 */
std::unique_ptr<InferenceBackend> createInferenceBackend(
    const std::string& backendName,
    const std::string& modelPath,
    float confidenceThreshold,
    float nmsThreshold,
    const cv::Size& inputSize = cv::Size(640, 640),
    const std::string& calibrationList = ""
);