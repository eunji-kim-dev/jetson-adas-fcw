#pragma once

#include "perception/Detection.hpp"

#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

// detect() 내부 단계별 소요 시간 (ms)
// 성능 로그가 필요 없는 호출자는 timing에 nullptr를 넘기면 됨
struct DetectionTiming {
    double fullYoloMilliseconds = 0.0;
    double farYoloMilliseconds = 0.0;
    double postprocessMilliseconds = 0.0;
};

/*
 * YOLOv8n ONNX 기반 객체 검출 파이프라인
 *
 * 전체 프레임 추론 + 원거리 crop 보조 추론 + 그룹 NMS +
 * 포함형 중복 제거 + 극소 박스 필터까지를 한 번에 수행
 *
 * 추론 엔진(cv::dnn::Net)은 private 멤버로 격리되어 있어
 * 이후 InferenceBackend 도입 시 이 클래스 내부만 교체하면 됨
 */
class YoloDetector {
public:
    YoloDetector(const std::string& modelPath, float confidenceThreshold, float nmsThreshold);

    std::vector<Detection> detect(const cv::Mat& frame, DetectionTiming* timing = nullptr);

private:
    // 단일 프레임(또는 crop)에 대한 YOLO 추론 + 클래스 그룹별 NMS
    std::vector<Detection> runInference(const cv::Mat& frame);
    // 화면 중앙 원거리 도로 영역 crop 보조 추론
    std::vector<Detection> detectFarRoadObjects(const cv::Mat& frame);

    cv::dnn::Net net_;
    float confidenceThreshold_;
    float nmsThreshold_;
};
