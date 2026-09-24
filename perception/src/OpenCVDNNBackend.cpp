#include "OpenCVDNNBackend.hpp"

#include "YoloCommon.hpp"

#include <chrono>
#include <stdexcept>
#include <vector>

OpenCVDNNBackend::OpenCVDNNBackend(const std::string& modelPath, float confidenceThreshold, float nmsThreshold, const cv::Size& inputSize)
    : confidenceThreshold_(confidenceThreshold), nmsThreshold_(nmsThreshold), inputSize_(inputSize) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> OpenCVDNNBackend::infer(const cv::Mat& image, InferenceTiming* timing) {
    // 전처리: letterbox + NCHW blob. 입력 크기는 생성자에서 받은 값(기본 640x640)
    const auto preprocessStart = std::chrono::steady_clock::now();
    const LetterboxResult prepared = letterbox(image, inputSize_);
    cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false);
    const auto preprocessEnd = std::chrono::steady_clock::now();

    // 추론: 입력 바인딩 + forward
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    const auto inferenceEnd = std::chrono::steady_clock::now();

    // 후처리: 출력 텐서 디코드 → 원본 좌표 복원 → 클래스 필터 → 그룹 NMS
    // 계산은 YoloCommon 에 있고 TensorRT 백엔드와 같은 코드를 씀
    if (output.dims != 3) throw std::runtime_error("지원하지 않는 YOLO 출력 차원");

    std::vector<Detection> detections = decodeYoloOutput(
        output.ptr<float>(), output.size[1], output.size[2],
        prepared, image.size(), confidenceThreshold_, nmsThreshold_);
    const auto postprocessEnd = std::chrono::steady_clock::now();

    if (timing != nullptr) {
        timing->preprocessMilliseconds = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        timing->inferenceMilliseconds = std::chrono::duration<double, std::milli>(inferenceEnd - preprocessEnd).count();
        timing->postprocessMilliseconds = std::chrono::duration<double, std::milli>(postprocessEnd - inferenceEnd).count();
    }

    return detections;
}