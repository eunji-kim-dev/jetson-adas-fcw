#pragma once

#include "perception/Detection.hpp"

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

    // hold/bonnet: 이번 프레임 관측이 비정상이라 샘플을 안 넣고 직전 단계를 그대로 돌려준 상태
    // WarningPolicy 는 이 프레임에서 배너 카운터를 올리지도 지우지도 않음
    bool observationHeld = false;
};

// --lead-rule hold / bonnet: classifyObservations() 가 트랙마다 내는 이번 프레임 관측 판정
// held   : 비정상 관측이지만 보존 한도(3프레임) 안 — 샘플·이력·단계를 그대로 두고 LeadSelector 는 referenceBox(마지막 정상 박스)로 기하를 잇는다
// bonnet : 보닛 모양이 한도를 넘어 확정 — 후보·분석 대상에서 뺀다
// reset  : 높이 급변이 한도를 넘음 — history 규칙의 기존 리셋과 같이 처음부터 다시 쌓는다
// 셋 다 false 면 정상 관측
struct ObservationState {
    int frame = -1;
    bool held = false;
    bool bonnet = false;
    bool reset = false;
    cv::Rect referenceBox;
};

class RiskAnalyzer {
public:
    // heightCollapseResetRatio: 연속 프레임 박스 높이 변화가 이 비율을 넘으면 그 트랙 이력을 초기화함
    // 0 이면 끔 (기존 동작). --lead-rule history 에서 0.40 으로 켬
    explicit RiskAnalyzer(
        double fps,
        int frameHeight,
        int historySize = 15,
        int staleFrameLimit = 60,
        float heightCollapseResetRatio = 0.0F,
        bool bonnetRule = false,
        bool holdRule = false
    );

    // hold/bonnet: 이번 프레임 관측을 트랙마다 분류함. LeadSelector::update() 와 update() 보다 먼저 부름
    // 두 규칙 다 꺼져 있으면 아무것도 안 하고 observationStates() 는 비어 있음
    void classifyObservations(const std::vector<TrackedObject>& trackedObjects, int currentFrame);
    const std::unordered_map<int, ObservationState>& observationStates() const { return observationStates_; }

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

        // hold/bonnet: 비정상 관측이 연속된 프레임 수. 정상 관측이 오면 0
        int heldFrames = 0;
        // 마지막 정상 관측 박스. held 프레임에 LeadSelector 가 기하를 이을 때 씀
        cv::Rect lastGoodBox;
        bool hasLastGoodBox = false;
        // classifyObservations() 가 이번 프레임에 낸 판정. update() 가 읽음
        ObservationState observation;
        // held 프레임의 샘플. 관측 불량이 끝나면 버리고, 한도를 넘겨 리셋할 때는 이걸로 새 이력을 시작함
        // (변화가 계속되는 경우 즉시 리셋하던 history 보다 3프레임 늦어지지 않게. 08 에서 확인)
        std::vector<Sample> heldSamples;
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

    // hold/bonnet 판정 보조. 상태는 바꾸지 않음
    float compensatedHeight(const TrackHistory& history, const cv::Rect& box) const;
    bool isHeightAnomaly(const TrackHistory& history, float boxHeight, int currentFrame) const;
    bool isBonnetShape(const cv::Rect& box) const;

    // 샘플은 남기고 단계 판정 상태만 SAFE로 되돌림
    static void clearLevelState(
        TrackHistory& history
    );

    RiskLevel stabilizeLevel(
        TrackHistory& history,
        RiskLevel rawLevel
    ) const;

    double fps_;
    int frameHeight_;
    std::size_t historySize_;
    int staleFrameLimit_;
    int maximumHistoryGapFrames_;
    float heightCollapseResetRatio_;
    bool bonnetRule_;
    bool holdRule_;

    std::unordered_map<int, TrackHistory> histories_;
    std::unordered_map<int, ObservationState> observationStates_;
};