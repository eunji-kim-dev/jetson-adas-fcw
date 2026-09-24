#pragma once

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

// 모델 입력 크기(가로 x 세로)에 맞춰 비율을 유지한 채 축소하고 회색(114)으로 패딩함
// 정사각형(640x640)이면 예전과 계산이 완전히 같음. 288x640 처럼 직사각형도 받음
LetterboxResult letterbox(const cv::Mat& frame, const cv::Size& inputSize);

/*
 * YOLOv8 출력 텐서를 Detection 목록으로 바꿈
 *
 * output      : 행 우선(row-major) float 버퍼. (1, 84, 8400) 이면 rows=84, cols=8400
 * rows, cols  : 배치 차원을 뺀 두 차원. rows < cols 면 안에서 전치해 행 하나가 후보 하나가 되게 함
 * prepared    : letterbox 결과 (원본 좌표 복원용)
 * imageSize   : 원본 이미지 크기 (좌표 클램프용)
 *
 * 디코드 → 원본 좌표 복원 → 클래스 필터 → 그룹 NMS 까지 처리함
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