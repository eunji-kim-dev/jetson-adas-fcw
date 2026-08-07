#pragma once

#include "Detection.hpp"

#include <cstddef>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>

enum class RiskLevel {
    Safe = 0,
    Caution = 1,
    Danger = 2
};

struct RiskResult {
    RiskLevel level = RiskLevel::Safe;
    bool valid = false;
    float ttcSeconds = std::numeric_limits<float>::infinity();
    float logHeightRatePerSecond = 0.0F;
    float groundSpeedPixelsPerSecond = 0.0F;
    float heightGrowthRatio = 1.0F;
    int sampleCount = 0;

    // bbox 하단이 화면 경계에 닿아 높이/접지점 측정이 포화된 상태
    bool truncated = false;
};

class RiskAnalyzer {
public:
    explicit RiskAnalyzer(
        double fps,
        int frameHeight,
        int historySize = 15,
        int staleFrameLimit = 60
    );

    RiskResult update(
        const TrackedObject& trackedObject,
        bool isAnalysisTarget,
        bool isLeadTarget,
        int currentFrame
    );

    // 장면 전환 시 모든 TTC-P 이력 제거
    void reset();

    void removeStaleTracks(int currentFrame);

    static std::string toString(RiskLevel level);

private:
    struct Sample {
        int frameIndex;
        float boxHeight;
        float groundY;
    };

    struct TrackHistory {
        std::deque<Sample> samples;
        int lastSeenFrame = -1;
        RiskLevel stableLevel = RiskLevel::Safe;
        RiskLevel pendingLevel = RiskLevel::Safe;
        int pendingFrames = 0;
        int lowerLevelFrames = 0;

        // 절단되지 않았던 마지막 프레임의 width / height
        // 하단 절단 후 width를 등가 height로 환산할 때 사용
        float lastAspectRatio = 0.0F;
    };

    static float calculateRegressionSlope(
        const std::deque<Sample>& samples,
        double fps,
        bool useLogHeight
    );

    static float calculateAverageHeight(
        const std::deque<Sample>& samples,
        std::size_t start,
        std::size_t count
    );

    static void resetTrackHistory(TrackHistory& history);

    RiskLevel stabilizeLevel(
        TrackHistory& history,
        RiskLevel rawLevel
    ) const;

    double fps_;
    int frameHeight_;
    std::size_t historySize_;
    int staleFrameLimit_;
    int maximumHistoryGapFrames_;

    std::unordered_map<int, TrackHistory> histories_;
};