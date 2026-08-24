#pragma once

#include "perception/Detection.hpp"

#include <opencv2/video/tracking.hpp>

#include <vector>

class MultiObjectTracker {
public:

    explicit MultiObjectTracker(
        float highConfidenceThreshold = 0.25F, // 새 Track 생성 및 1차 매칭에 사용하는 신뢰도 기준
        float lowConfidenceThreshold = 0.10F, // 기존 Track 유지에 사용할 최저 신뢰도 기준
        int minConfirmationHits = 3, // 몇 프레임 연속 검출돼야 정식 ID를 발급할지 결정
        int maxMissedFrames = 20 // 확정된 Track을 최대 몇 프레임까지 보관할지 결정
    );

    std::vector<TrackedObject> update(
        const std::vector<Detection>& detections
    );

    // 장면 전환 시 기존 Track/Kalman 상태만 제거
    // nextTrackId_는 유지해서 ID를 재사용하지 않음
    void reset();

private:
    struct TrackState {
        
        // 확정되기 전에는 id가 0
        // 순간적인 오검출은 정식 ID 번호를 소모하지 않음
        int id;
        int classId;
        float confidence;

        cv::Rect2f box;
        cv::KalmanFilter kalman;

        int age;
        int hits;
        int consecutiveHits;
        int missedFrames;

        bool confirmed;

        explicit TrackState(
            const Detection& detection
        );
    };

    static float calculateIoU(
        const cv::Rect2f& first,
        const cv::Rect2f& second
    );

    static float calculateCenterDistance(
        const cv::Rect2f& first,
        const cv::Rect2f& second
    );

    static float calculateAreaRatio(
        const cv::Rect2f& first,
        const cv::Rect2f& second
    );

    static float calculateAllowedCenterDistance(
        const cv::Rect2f& first,
        const cv::Rect2f& second
    );

    static cv::Rect2f stateToRect(
        const cv::Mat& state
    );

    static cv::Mat detectionToMeasurement(
        const Detection& detection
    );

    static bool isSameTrackingCategory(
        int firstClassId,
        int secondClassId
    );

    float highConfidenceThreshold_;
    float lowConfidenceThreshold_;

    int minConfirmationHits_;
    int maxMissedFrames_;
    int nextTrackId_;

    std::vector<TrackState> tracks_;
};