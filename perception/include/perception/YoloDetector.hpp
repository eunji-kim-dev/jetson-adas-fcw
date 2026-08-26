#pragma once

#include "perception/Detection.hpp"
#include "perception/InferenceBackend.hpp"

#include <opencv2/core.hpp>
#include <memory>
#include <vector>

// detect() 내부 단계별 소요 시간 (ms)
// 성능 로그가 필요 없는 호출자는 timing에 nullptr를 넘기면 됨
//
// - fullYoloMilliseconds : 전체 프레임 backend 호출 전체 (전처리+추론+디코드/NMS)
// - farYoloMilliseconds  : 원거리 crop 생성 + backend 호출 + 필터/좌표 복원
// - postprocessMilliseconds : 두 결과 병합 NMS + 포함형 중복 제거 + 극소 박스 필터
// - fullInference / farInference : 각 backend 호출 내부의 단계별 시간
//   (crop 영역이 유효하지 않아 추론을 건너뛰면 farInference는 0으로 남음)
struct DetectionTiming {
    double fullYoloMilliseconds = 0.0;
    double farYoloMilliseconds = 0.0;
    double postprocessMilliseconds = 0.0;
    InferenceTiming fullInference;
    InferenceTiming farInference;
};

/*
 * YOLOv8 기반 객체 검출 파이프라인
 *
 * 전체 프레임 추론 + 원거리 crop 보조 추론 + 그룹 NMS +
 * 포함형 중복 제거 + 극소 박스 필터까지를 한 번에 수행
 *
 * 실제 추론(전처리/forward/디코드)은 InferenceBackend에 위임하고,
 * 이 클래스는 "어떤 이미지를 몇 번 추론하고 결과를 어떻게 합칠지"만 담당한다.
 * 백엔드는 자신이 받은 이미지가 전체 프레임인지 crop인지 알지 못한다.
 */
class YoloDetector {
public:
    YoloDetector(std::unique_ptr<InferenceBackend> backend, float nmsThreshold);

    std::vector<Detection> detect(const cv::Mat& frame, DetectionTiming* timing = nullptr);

private:
    // 화면 중앙 원거리 도로 영역 crop 보조 추론
    std::vector<Detection> detectFarRoadObjects(const cv::Mat& frame, InferenceTiming* timing);

    std::unique_ptr<InferenceBackend> backend_;
    float nmsThreshold_;
};
