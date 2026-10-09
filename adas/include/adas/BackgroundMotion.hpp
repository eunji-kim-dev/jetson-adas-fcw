#pragma once

#include "perception/Detection.hpp"

#include <opencv2/core.hpp>
#include <vector>

// 직전 처리 프레임에서 현재 프레임까지의 배경 공통 수평 이동임. 실제 yaw 각도는 아님
struct BackgroundMotionResult {
    bool valid = false;
    float dxPixels = 0.0F;
    int leftCount = 0;
    int rightCount = 0;
    float leftDxPixels = 0.0F;
    float rightDxPixels = 0.0F;
    const char* state = "first"; // 추정 결과 상태임. zero는 중심 수평 이동만 0 근처라는 뜻임
};

class BackgroundMotionEstimator {
public:
    // 박스와 글자를 그리기 전 원본 프레임을 받음. 이동량은 원본 영상 픽셀 단위임
    BackgroundMotionResult update(const cv::Mat& frame,
                                  const std::vector<Detection>& detections,
                                  const std::vector<TrackedObject>& trackedObjects);
    void reset();

private:
    cv::Mat previousGray_;
    cv::Mat previousMask_;
    cv::Size previousSourceSize_;
};