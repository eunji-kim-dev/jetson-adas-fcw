#include "YoloCommon.hpp"

#include "GroupedNms.hpp"
#include "perception/Classes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

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

std::vector<Detection> decodeYoloOutput(
    const float* output,
    int rows,
    int cols,
    const LetterboxResult& prepared,
    const cv::Size& imageSize,
    float confidenceThreshold,
    float nmsThreshold
) {
    cv::Mat predictions(rows, cols, CV_32F, const_cast<float*>(output));

    // (84, 8400) 형태면 (8400, 84)로 전치해 행 하나가 후보 하나가 되게 함
    if (rows < cols) predictions = predictions.t();

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

        if (rawConfidence < confidenceThreshold) continue;

        int left = static_cast<int>(std::round((centerX - boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int top = static_cast<int>(std::round((centerY - boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));
        int right = static_cast<int>(std::round((centerX + boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int bottom = static_cast<int>(std::round((centerY + boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));

        left = std::clamp(left, 0, imageSize.width - 1);
        top = std::clamp(top, 0, imageSize.height - 1);
        right = std::clamp(right, 0, imageSize.width - 1);
        bottom = std::clamp(bottom, 0, imageSize.height - 1);

        if (right <= left || bottom <= top) continue;

        const int restoredWidth = right - left;
        const int restoredHeight = bottom - top;

        if (!isTargetClass(rawClassId)) continue;

        candidates.push_back({rawClassId, rawConfidence, cv::Rect(left, top, restoredWidth, restoredHeight)});
    }

    // 단일 이미지 내 중복 박스 제거
    // NMSBoxes 규칙상 confidence > confidenceThreshold 인 후보만 유지됨
    return applyGroupedNms(candidates, confidenceThreshold, nmsThreshold);
}