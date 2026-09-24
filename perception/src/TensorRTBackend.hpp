#pragma once

#include "perception/InferenceBackend.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

/*
 * TensorRT(GPU) 기반 YOLOv8 추론 백엔드
 *
 * 이 헤더는 perception/src 내부 전용임.
 * TensorRT / CUDA 타입은 전부 cpp 안에 숨겨서
 * 이 헤더를 include 하는 쪽이 NvInfer.h 를 몰라도 되게 함.
 *
 * 정밀도
 *   - "fp32" : 기본. OpenCV DNN 과 가장 가까운 결과
 *   - "fp16" : 절반 정밀도. Orin 에서 2~3배 빠르지만 점수가 미세하게 달라짐
 *   - "int8" : 8비트 정수. Entropy calibration 으로 층별 스케일을 정함.
 *              INT8 이 안 되는 층은 FP16 으로 떨어짐 (kFP16 도 같이 켬).
 *              처음 엔진을 만들 때 calibration 이미지 목록이 필요함.
 *
 * 엔진 캐시
 *   ONNX → 엔진 빌드는 Jetson 에서 수 분 걸리므로
 *   <모델경로>.<정밀도>.engine 에 저장해 두고 다음부터 그대로 읽음.
 *   엔진이 없거나, ONNX 가 엔진보다 새거나, 읽기에 실패하면 다시 빌드함.
 *   엔진 파일은 GPU 와 TensorRT 버전에 묶여 있어 다른 장비에서는 못 씀.
 *
 * Calibration 캐시 (int8 만)
 *   <모델경로>.int8.calib 에 층별 스케일을 저장함. 이 파일이 있으면 이미지를 다시 안 읽음.
 *   calibration 이미지를 바꾸면 이 파일을 지워야 새로 계산함.
 */
class TensorRTBackend : public InferenceBackend {
public:
    // expectedInputSize 는 호출자가 기대하는 입력 크기(가로 x 세로).
    // 엔진에서 읽은 크기와 다르면 예외. 엔진 캐시가 다른 모델 것일 때 조용히 틀리는 걸 막음
    // calibrationList 는 int8 에서 엔진을 새로 만들 때 쓰는 이미지 목록 파일. 그 외에는 무시함
    TensorRTBackend(const std::string& modelPath, const std::string& precision, float confidenceThreshold, float nmsThreshold,
                    const cv::Size& expectedInputSize, const std::string& calibrationList = "");
    ~TensorRTBackend() override;

    TensorRTBackend(const TensorRTBackend&) = delete;
    TensorRTBackend& operator=(const TensorRTBackend&) = delete;

    std::vector<Detection> infer(const cv::Mat& image, InferenceTiming* timing) override;

    // 실제로 쓰인 엔진 파일 경로와 빌드 여부 (run_summary 기록용)
    const std::string& enginePath() const { return enginePath_; }
    bool engineWasBuilt() const { return engineWasBuilt_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string enginePath_;
    bool engineWasBuilt_ = false;

    float confidenceThreshold_;
    float nmsThreshold_;
};