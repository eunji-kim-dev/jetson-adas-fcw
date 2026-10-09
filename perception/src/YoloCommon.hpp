#pragma once

#include "YoloCandidate.hpp"
#include "perception/Detection.hpp"

#include <opencv2/core.hpp>
#include <vector>

/*
 * YOLOv8 전처리·후처리 공용 코드
 *
 * 이 헤더는 perception/src 내부 전용임.
 * 추론 엔진(OpenCV DNN, TensorRT)이 달라도 letterbox 와 출력 디코드는
 * 같은 계산을 써야 백엔드 간 결과 차이가 엔진 차이만 반영함.
 */

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

// letterbox 크기 계산 (축소 비율·resize 크기·여백). CPU letterbox 와 GPU 전처리가 같은 값을 쓰게 한 곳에 둠
struct LetterboxGeometry {
    float scale;
    cv::Size resized;
    int padLeft;
    int padTop;
    int padRight;
    int padBottom;
};
LetterboxGeometry computeLetterboxGeometry(const cv::Size& frameSize, const cv::Size& inputSize);

// CPU 세부 타이머: letterbox 안의 resize·pad 시간 (ms)
struct LetterboxTiming {
    double resizeMilliseconds = 0.0;
    double padMilliseconds = 0.0;
};

// 모델 입력 크기(가로 x 세로)에 맞춰 비율을 유지한 채 축소하고 회색(114)으로 패딩함
// 정사각형(640x640)이면 예전과 계산이 완전히 같음. 288x640 처럼 직사각형도 받음
// timing 이 nullptr 가 아니면 resize·pad 시간을 채움
LetterboxResult letterbox(const cv::Mat& frame, const cv::Size& inputSize, LetterboxTiming* timing = nullptr);

/*
 * YOLOv8 출력 텐서를 Detection 목록으로 바꿈
 *
 * output      : 행 우선(row-major) float 버퍼. (1, 84, 8400) 이면 rows=84, cols=8400
 * rows, cols  : 배치 차원을 뺀 두 차원. rows < cols 면 안에서 전치해 행 하나가 후보 하나가 되게 함
 * prepared    : letterbox 결과 (원본 좌표 복원용)
 * imageSize   : 원본 이미지 크기 (좌표 클램프용)
 *
 * 디코드 → 원본 좌표 복원 → 클래스 필터 → 그룹 NMS 까지 처리함
 * = scanYoloOutput (1단계) + restoreYoloCandidates (2단계). 결과는 나누기 전과 같음 (golden 유지)
 */
std::vector<Detection> decodeYoloOutput(
    const float* output,
    int rows,
    int cols,
    const LetterboxResult& prepared,
    const cv::Size& imageSize,
    float confidenceThreshold,
    float nmsThreshold
);

/*
 * decodeYoloOutput 의 두 단계. GPU 후처리(--gpu-postprocess)는 1단계만 GPU 커널로 하고 2단계는 같은 CPU 코드를 씀
 *   1단계 scanYoloOutput        : 후보마다 클래스 최댓값(cv::minMaxLoc) → 문턱(score < threshold 면 버림) → YoloCandidate 목록 (후보 번호 순)
 *   2단계 restoreYoloCandidates : 박스 좌표 복원 → 클램프 → 빈 박스·대상 클래스 필터 → 그룹 NMS
 *
 * yoloOutputLayout         : (rows, cols) 가 (C, N) 인지 (N, C) 인지 보고 읽는 자리를 정함 (rows < cols 면 (C, N). decodeYoloOutput 의 전치 조건과 같음)
 * scanYoloOutputReference  : 1단계를 GPU 커널과 같은 함수(scanYoloCandidate)로 CPU 에서 한 것. --gpu-postprocess-check 비교용
 * gatherYoloCandidates     : 커널 결과(flags·records, 길이 columns)를 후보 번호 순 목록으로 모음
 */
YoloOutputLayout yoloOutputLayout(int rows, int cols);
std::vector<YoloCandidate> scanYoloOutput(const float* output, int rows, int cols, float confidenceThreshold);
std::vector<YoloCandidate> scanYoloOutputReference(const float* output, int rows, int cols, float confidenceThreshold);
std::vector<YoloCandidate> gatherYoloCandidates(const unsigned char* flags, const YoloCandidate* records, int columns);
std::vector<Detection> restoreYoloCandidates(
    const std::vector<YoloCandidate>& candidates,
    const LetterboxResult& prepared,
    const cv::Size& imageSize,
    float confidenceThreshold,
    float nmsThreshold
);