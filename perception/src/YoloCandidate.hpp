#pragma once

#include <cstddef>

/*
 * YOLOv8 출력 텐서의 후보 한 칸 훑기 — CPU 기준 함수(YoloCommon.cpp)와 GPU 커널(GpuPostprocess.cu)이 같은 함수를 씀
 *
 * 출력 텐서 (1, C, N): C = 4(박스 cx·cy·w·h) + 80(클래스 점수), N = 후보 수 (640x640 은 8400, 288x640 은 3780)
 * 후보 하나의 일 = 클래스 점수 80개에서 최댓값과 그 위치를 찾고 문턱과 비교하는 것뿐임 → 실수 계산 없이 비교만 함
 * 박스 좌표 복원·클램프·클래스 필터·NMS 는 그 뒤 CPU 가 기존 코드로 함 (YoloCommon.cpp restoreYoloCandidates)
 *
 * 최댓값 규칙은 cv::minMaxLoc 과 같음: 같은 값이면 앞 인덱스, NaN 은 건너뜀 (NaN 은 어느 비교도 참이 아님)
 * 80개 전부 NaN 이면 cv::minMaxLoc 처럼 인덱스 0·값 0 으로 둠 (문턱에 걸려 버려짐)
 * 이 파일은 OpenCV·CUDA 없이 g++·nvcc 둘 다에서 컴파일됨
 */

#ifdef __CUDACC__
#define YOLO_HOST_DEVICE __host__ __device__
#else
#define YOLO_HOST_DEVICE
#endif

// 문턱을 넘은 후보 하나. 값은 출력 텐서의 float 를 그대로 옮긴 것 (비트 단위로 같음)
struct YoloCandidate {
    int column;        // 후보 번호 (0 ~ N-1). 이 순서가 NMS 의 동점 순서를 정하므로 목록은 항상 이 순서로 둠
    int classIndex;    // 최댓값 클래스 (0 ~ 79 = COCO 클래스 번호)
    float score;       // 그 클래스 점수
    float centerX;     // 박스 가운데 x (모델 입력 좌표)
    float centerY;
    float width;
    float height;
};

// 출력 텐서에서 값 (channel, column) 을 읽는 자리: output[channel * channelStride + column * columnStride]
// (1, C, N) 이면 channelStride = N, columnStride = 1. (1, N, C) 면 channelStride = 1, columnStride = C
struct YoloOutputLayout {
    int channels = 0;                // C (4 + 클래스 수)
    int columns = 0;                 // N (후보 수)
    std::size_t channelStride = 0;
    std::size_t columnStride = 0;
};

// 후보 column 하나를 훑음. 문턱을 넘으면(score < threshold 가 아니면) out 을 채우고 true
YOLO_HOST_DEVICE inline bool scanYoloCandidate(const float* output, const YoloOutputLayout& layout, int column, float threshold,
                                               YoloCandidate& out) {
    const float* base = output + static_cast<std::size_t>(column) * layout.columnStride;
    int bestIndex = -1;
    float bestScore = 0.0F;
    for (int channel = 4; channel < layout.channels; ++channel) {
        const float value = base[static_cast<std::size_t>(channel) * layout.channelStride];
        if (bestIndex < 0) {
            // 처음 만난 값(NaN 이 아닌 것)으로 시작함
            if (value == value) {
                bestScore = value;
                bestIndex = channel - 4;
            }
        } else if (value > bestScore) {
            // 더 큰 값일 때만 바꿈 → 같은 값이면 앞 인덱스가 남음 (cv::minMaxLoc 과 같음)
            bestScore = value;
            bestIndex = channel - 4;
        }
    }
    if (bestIndex < 0) {
        bestIndex = 0;
        bestScore = 0.0F;
    }
    if (bestScore < threshold) return false;

    out.column = column;
    out.classIndex = bestIndex;
    out.score = bestScore;
    out.centerX = base[0];
    out.centerY = base[static_cast<std::size_t>(1) * layout.channelStride];
    out.width = base[static_cast<std::size_t>(2) * layout.channelStride];
    out.height = base[static_cast<std::size_t>(3) * layout.channelStride];
    return true;
}