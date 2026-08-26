#include "OpenCVDNNBackend.hpp"

#include "GroupedNms.hpp"
#include "perception/Classes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

// YOLOv8 입력 크기(정사각형)에 맞춰 비율을 유지한 채 축소하고 회색(114)으로 패딩
LetterboxResult letterbox(const cv::Mat& frame, int inputSize) {
    const float scale = std::min(static_cast<float>(inputSize) / static_cast<float>(frame.cols), static_cast<float>(inputSize) / static_cast<float>(frame.rows));
    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(frame.cols) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(frame.rows) * scale));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight));

    const int totalPadX = inputSize - resizedWidth;
    const int totalPadY = inputSize - resizedHeight;
    const int padLeft = totalPadX / 2;
    const int padRight = totalPadX - padLeft;
    const int padTop = totalPadY / 2;
    const int padBottom = totalPadY - padTop;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return {padded, scale, padLeft, padTop};
}

} // namespace

OpenCVDNNBackend::OpenCVDNNBackend(const std::string& modelPath, float confidenceThreshold, float nmsThreshold)
    : confidenceThreshold_(confidenceThreshold), nmsThreshold_(nmsThreshold) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> OpenCVDNNBackend::infer(const cv::Mat& image, InferenceTiming* timing) {
    constexpr int inputSize = 640;

    // 전처리: letterbox + NCHW blob
    const auto preprocessStart = std::chrono::steady_clock::now();
    const LetterboxResult prepared = letterbox(image, inputSize);
    cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, cv::Size(inputSize, inputSize), cv::Scalar(), true, false);
    const auto preprocessEnd = std::chrono::steady_clock::now();

    // 추론: 입력 바인딩 + forward
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    const auto inferenceEnd = std::chrono::steady_clock::now();

    // 후처리: 출력 텐서 디코드 → 원본 좌표 복원 → 클래스 필터 → 그룹 NMS
    if (output.dims != 3) throw std::runtime_error("지원하지 않는 YOLO 출력 차원");

    const int firstDimension = output.size[1];
    const int secondDimension = output.size[2];
    cv::Mat predictions(firstDimension, secondDimension, CV_32F, output.ptr<float>());

    // (1, 84, 8400) 형태면 (8400, 84)로 전치해 행 하나가 후보 하나가 되게 함
    if (firstDimension < secondDimension) predictions = predictions.t();

    std::vector<Detection> candidates;

    for (int row = 0; row < predictions.rows; ++row) {
        const float* data = predictions.ptr<float>(row);
        const float centerX = data[0], centerY = data[1], boxWidth = data[2], boxHeight = data[3];

        cv::Mat classScores(1, predictions.cols - 4, CV_32F, const_cast<float*>(data + 4));
        cv::Point bestClassPoint;
        double bestClassScore = 0.0;
        cv::minMaxLoc(classScores, nullptr, &bestClassScore, nullptr, &bestClassPoint);

        const int rawClassId = bestClassPoint.x;
        const float rawConfidence = static_cast<float>(bestClassScore);

        if (rawConfidence < confidenceThreshold_) continue;

        int left = static_cast<int>(std::round((centerX - boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int top = static_cast<int>(std::round((centerY - boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));
        int right = static_cast<int>(std::round((centerX + boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int bottom = static_cast<int>(std::round((centerY + boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));

        left = std::clamp(left, 0, image.cols - 1);
        top = std::clamp(top, 0, image.rows - 1);
        right = std::clamp(right, 0, image.cols - 1);
        bottom = std::clamp(bottom, 0, image.rows - 1);

        if (right <= left || bottom <= top) continue;

        const int restoredWidth = right - left;
        const int restoredHeight = bottom - top;

        if (!isTargetClass(rawClassId)) continue;

        candidates.push_back({rawClassId, rawConfidence, cv::Rect(left, top, restoredWidth, restoredHeight)});
    }

    // 단일 이미지 내 중복 박스 제거
    // NMSBoxes 규칙상 confidence > confidenceThreshold_ 인 후보만 유지됨
    std::vector<Detection> detections = applyGroupedNms(candidates, confidenceThreshold_, nmsThreshold_);
    const auto postprocessEnd = std::chrono::steady_clock::now();

    if (timing != nullptr) {
        timing->preprocessMilliseconds = std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        timing->inferenceMilliseconds = std::chrono::duration<double, std::milli>(inferenceEnd - preprocessEnd).count();
        timing->postprocessMilliseconds = std::chrono::duration<double, std::milli>(postprocessEnd - inferenceEnd).count();
    }

    return detections;
}
