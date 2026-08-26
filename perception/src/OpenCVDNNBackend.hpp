#pragma once

#include "perception/InferenceBackend.hpp"

#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

/*
 * OpenCV DNN(CPU) 기반 YOLOv8 ONNX 추론 백엔드
 *
 * 이 헤더는 perception/src 내부 전용이다.
 * 멤버로 cv::dnn::Net을 가지므로 공개 헤더(include/)에 두면
 * opencv_dnn 의존성이 상위 계층으로 새어 나간다.
 * 상위 계층은 createInferenceBackend("opencv_dnn", ...)로만 생성한다.
 */
class OpenCVDNNBackend : public InferenceBackend {
public:
    OpenCVDNNBackend(const std::string& modelPath, float confidenceThreshold, float nmsThreshold);

    std::vector<Detection> infer(const cv::Mat& image, InferenceTiming* timing) override;

private:
    cv::dnn::Net net_;
    float confidenceThreshold_;
    float nmsThreshold_;
};
