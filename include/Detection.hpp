#pragma once

#include <opencv2/core.hpp>

// YOLO가 현재 프레임에서 검출한 객체 하나
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

// 여러 프레임에 걸쳐 추적되고 있는 객체 하나
struct TrackedObject {
    int trackId;
    int classId;
    float confidence;
    cv::Rect box;
};