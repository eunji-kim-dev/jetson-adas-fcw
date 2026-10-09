#include "YoloCommon.hpp"

#include "GroupedNms.hpp"
#include "perception/Classes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>

LetterboxGeometry computeLetterboxGeometry(const cv::Size& frameSize, const cv::Size& inputSize) {
    // 가로·세로 축소 비율 중 작은 쪽을 씀. 입력이 정사각형이면 예전 계산과 같음
    const float scale = std::min(static_cast<float>(inputSize.width) / static_cast<float>(frameSize.width), static_cast<float>(inputSize.height) / static_cast<float>(frameSize.height));
    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(frameSize.width) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(frameSize.height) * scale));

    // 여백은 가로·세로 따로 계산함 (직사각형 입력 대응)
    const int totalPadX = inputSize.width - resizedWidth;
    const int totalPadY = inputSize.height - resizedHeight;
    const int padLeft = totalPadX / 2;
    const int padRight = totalPadX - padLeft;
    const int padTop = totalPadY / 2;
    const int padBottom = totalPadY - padTop;

    return {scale, cv::Size(resizedWidth, resizedHeight), padLeft, padTop, padRight, padBottom};
}

LetterboxResult letterbox(const cv::Mat& frame, const cv::Size& inputSize, LetterboxTiming* timing) {
    const LetterboxGeometry geometry = computeLetterboxGeometry(frame.size(), inputSize);

    const auto resizeStart = std::chrono::steady_clock::now();
    cv::Mat resized;
    cv::resize(frame, resized, geometry.resized);
    const auto resizeEnd = std::chrono::steady_clock::now();

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, geometry.padTop, geometry.padBottom, geometry.padLeft, geometry.padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    const auto padEnd = std::chrono::steady_clock::now();

    if (timing != nullptr) {
        timing->resizeMilliseconds = std::chrono::duration<double, std::milli>(resizeEnd - resizeStart).count();
        timing->padMilliseconds = std::chrono::duration<double, std::milli>(padEnd - resizeEnd).count();
    }

    return {padded, geometry.scale, geometry.padLeft, geometry.padTop};
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
    // 1단계(클래스 최댓값·문턱) → 2단계(좌표 복원·필터·NMS). 나누기 전과 같은 계산 순서임
    return restoreYoloCandidates(scanYoloOutput(output, rows, cols, confidenceThreshold), prepared, imageSize, confidenceThreshold, nmsThreshold);
}

YoloOutputLayout yoloOutputLayout(int rows, int cols) {
    YoloOutputLayout layout;
    if (rows < cols) {
        // (C, N): 채널이 바깥, 후보가 안쪽 (TensorRT·OpenCV DNN 의 YOLOv8 출력 (1, 84, 8400))
        layout.channels = rows;
        layout.columns = cols;
        layout.channelStride = static_cast<std::size_t>(cols);
        layout.columnStride = 1;
    } else {
        // (N, C): 후보가 바깥
        layout.channels = cols;
        layout.columns = rows;
        layout.channelStride = 1;
        layout.columnStride = static_cast<std::size_t>(cols);
    }
    return layout;
}

// 1단계 (CPU 경로, golden): 후보마다 cv::minMaxLoc 으로 클래스 최댓값을 찾고 문턱을 넘은 것만 남김
std::vector<YoloCandidate> scanYoloOutput(const float* output, int rows, int cols, float confidenceThreshold) {
    cv::Mat predictions(rows, cols, CV_32F, const_cast<float*>(output));

    // (84, 8400) 형태면 (8400, 84)로 전치해 행 하나가 후보 하나가 되게 함
    if (rows < cols) predictions = predictions.t();

    std::vector<YoloCandidate> candidates;

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

        candidates.push_back({row, rawClassId, rawConfidence, centerX, centerY, boxWidth, boxHeight});
    }
    return candidates;
}

// 1단계 기준 함수: GPU 커널과 같은 scanYoloCandidate() 를 후보 순서대로 CPU 에서 돌림 (--gpu-postprocess-check 비교용)
std::vector<YoloCandidate> scanYoloOutputReference(const float* output, int rows, int cols, float confidenceThreshold) {
    const YoloOutputLayout layout = yoloOutputLayout(rows, cols);
    std::vector<YoloCandidate> candidates;
    for (int column = 0; column < layout.columns; ++column) {
        YoloCandidate candidate;
        if (scanYoloCandidate(output, layout, column, confidenceThreshold, candidate)) candidates.push_back(candidate);
    }
    return candidates;
}

// 커널 결과 모으기: flags 가 1 인 후보의 records 만 후보 번호 순으로
std::vector<YoloCandidate> gatherYoloCandidates(const unsigned char* flags, const YoloCandidate* records, int columns) {
    std::vector<YoloCandidate> candidates;
    for (int column = 0; column < columns; ++column) {
        if (flags[column]) candidates.push_back(records[column]);
    }
    return candidates;
}

// 2단계: 박스 좌표 복원 → 클램프 → 빈 박스·대상 클래스 필터 → 그룹 NMS (나누기 전 decodeYoloOutput 의 나머지 반)
std::vector<Detection> restoreYoloCandidates(
    const std::vector<YoloCandidate>& scanned,
    const LetterboxResult& prepared,
    const cv::Size& imageSize,
    float confidenceThreshold,
    float nmsThreshold
) {
    std::vector<Detection> candidates;

    for (const YoloCandidate& scan : scanned) {
        const float centerX = scan.centerX, centerY = scan.centerY, boxWidth = scan.width, boxHeight = scan.height;
        const int rawClassId = scan.classIndex;
        const float rawConfidence = scan.score;

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