#pragma once

#include "perception/Detection.hpp"
#include "adas/RoadHomography.hpp"

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

    // --diag-log 진단용. 판정에는 안 씀
    // rawLevel     : 안정화(stabilizeLevel) 전 이번 프레임 단계. LEAD 가 아닌 트랙도 "LEAD 였다면" 값을 채움
    // boxHeight    : 샘플에 넣은 절단 보정 높이 (held 면 샘플에 안 넣은 이번 관측값). 0 이면 계산 안 함
    // groundY      : 샘플에 넣은 접지점 y (절단이면 가상 접지점). 0 이면 계산 안 함
    // aspectRatio  : 절단 보정에 쓰는 마지막 비절단 종횡비 (폭 ÷ 높이). 0 이면 아직 없음
    // historyReset : 이번 프레임에 샘플을 지운 이유 (gap | not_target | bonnet | hold_limit | anomaly). 비어 있으면 안 지움
    RiskLevel rawLevel = RiskLevel::Safe;
    float boxHeight = 0.0F;
    float groundY = 0.0F;
    float aspectRatio = 0.0F;
    const char* historyReset = "";

    // --ttc-mode homography|both: TTR-H 분기. TTC-P 와 상태를 공유하지 않음. 꺼져 있으면 아래는 기본값 그대로
    // hState       : 이번 프레임 H 상태. 비어 있으면 계산 안 함 (분석 대상 아님 등)
    //                excluded(아래 변이 화면 끝·보닛 선 근처) | horizon(지평선 위·근처) | bottom_drop(박스 높이가 직전 유효 샘플의 85% 미만 = 아래 변 누락 의심)
    //                | passed(Z_now ≤ Z_c) | few(거리 샘플 10개 미만) | receding(기울기 ≥ 0) | noisy(줄어든 거리 < 잔차 × 3) | valid
    //                | held(hold/bonnet 보존)
    // hObserved    : 이번 프레임 접지점이 유효해 거리 샘플을 넣었는지
    // hValid       : TTR-H 를 계산함 (hState == valid)
    // distanceM    : 이번 프레임 접지점 깊이 Z_now (m). hObserved 일 때만
    // hSpeedMps    : 거리 회귀 기울기 v̂_Z (m/s, 접근이면 음수). 거리 샘플 2개 이상일 때만
    // hResidualM   : 회귀 잔차 흔들림 (m). 거리 샘플 3개 이상일 때만
    // ttrH         : (Z_now − Z_c) / (−v̂_Z) 초. hValid 일 때만 유한
    // hSampleCount : 거리 샘플 수 (박스 샘플과 따로 셈)
    // hRawLevel    : 안정화 전 H 단계. LEAD 가 아닌 트랙도 "LEAD 였다면" 값 (rawLevel 과 같음)
    // hLevel       : 안정화된 H 단계. LEAD 만. hValid 가 아니면 SAFE 이고 안정화 상태도 지움 (새 경고 후보 누적을 끊음. 확정된 배너 유지는 WarningPolicy 몫)
    //                단 hold/bonnet 보존 프레임(held)은 직전 H 단계 그대로 (P 와 같음. 그 프레임은 배너 카운터가 동결됨)
    const char* hState = "";
    bool hObserved = false;
    bool hValid = false;
    float distanceM = 0.0F;
    float hSpeedMps = 0.0F;
    float hResidualM = 0.0F;
    float ttrH = std::numeric_limits<float>::infinity();
    int hSampleCount = 0;
    RiskLevel hRawLevel = RiskLevel::Safe;
    RiskLevel hLevel = RiskLevel::Safe;
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
    // bottomRule: --lead-rule bottom. 화면 안에서 아래 변만 놓친 프레임을 절단과 같이 폭 ÷ 종횡비로 보정함
    explicit RiskAnalyzer(
        double fps,
        int frameHeight,
        int historySize = 15,
        int staleFrameLimit = 60,
        float heightCollapseResetRatio = 0.0F,
        bool bonnetRule = false,
        bool holdRule = false,
        bool bottomRule = false
    );

    // hold/bonnet: 이번 프레임 관측을 트랙마다 분류함. LeadSelector::update() 와 update() 보다 먼저 부름
    // 두 규칙 다 꺼져 있으면 아무것도 안 하고 observationStates() 는 비어 있음
    void classifyObservations(const std::vector<TrackedObject>& trackedObjects, int currentFrame);
    const std::unordered_map<int, ObservationState>& observationStates() const { return observationStates_; }

    // bonnet: 보닛 반사 의심 모양인지 (아래 변 ≥ 95%H, 위 변 ≥ 55%H, 폭÷높이 ≥ 2.5). 상태는 안 바꿈
    // 결과 영상에서 보닛 박스를 안 그릴 때도 씀
    bool isBonnetShape(const cv::Rect& box) const;

    RiskResult update(
        const TrackedObject& trackedObject,
        bool isAnalysisTarget,
        bool isLeadTarget,
        int currentFrame
    );

    // --ttc-mode homography|both: 노면 변환표를 줌. 안 주면 TTR-H 분기는 안 돎 (기존 동작)
    void setRoadHomography(const RoadHomography& homography);
    bool homographyEnabled() const { return homographyEnabled_; }

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

    // TTR-H 거리 샘플 한 칸 (접지점 깊이 Z, m). boxHeight 는 아래 변 누락 판단용 박스 높이 (px)
    struct DistanceSample {
        int frameIndex;
        double depthM;
        int boxHeight;
    };

    // TTR-H 안정화 상태. TrackHistory 의 stableLevel·pendingLevel·pendingFrames·lowerLevelFrames(P) 와 같은 뜻으로 따로 둠
    struct LevelState {
        RiskLevel stableLevel = RiskLevel::Safe;
        RiskLevel pendingLevel = RiskLevel::Safe;
        int pendingFrames = 0;
        int lowerLevelFrames = 0;
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
        // bottom: lastAspectRatio 를 잡은 프레임의 박스 높이. 높이가 이보다 줄었을 때만 누락으로 봄 (폭이 넓어져 종횡비가 뛴 경우는 제외)
        float lastAspectHeight = 0.0F;
        // bottom: 아래 변 누락으로 보정한 프레임이 연속된 수. 정상 모양이 오면 0. 한도를 넘으면 새 모양으로 받아들임
        int bottomMissingFrames = 0;
        // 마지막 정상 박스의 윗변 y. 아래 변 누락은 윗변이 그대로일 때만 인정함
        float lastAspectTop = 0.0F;

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

        // TTR-H: 거리 샘플(최근 15개·1.0초 이내)과 안정화 상태. P 의 리셋(resetTrackHistory)에 같이 지워짐
        std::deque<DistanceSample> distanceSamples;
        LevelState homographyLevel;
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
    // bottom: 화면 안에서 아래 변만 놓친 박스인지 (절단 아님 + 높이가 마지막 정상 프레임보다 줄었음 + 폭÷높이가 그때보다 15% 넘게 큼 + 연속 한도 안)
    //         상태는 바꾸지 않음
    bool isBottomMissing(const TrackHistory& history, const cv::Rect& box) const;

    // 샘플은 남기고 단계 판정 상태만 SAFE로 되돌림
    static void clearLevelState(
        TrackHistory& history
    );

    RiskLevel stabilizeLevel(
        TrackHistory& history,
        RiskLevel rawLevel
    ) const;

    // TTR-H 안정화. stabilizeLevel 과 같은 규칙(올림 DANGER 2·CAUTION 4, 내림 DANGER 10·CAUTION 6)을 H 상태에 적용함
    static RiskLevel stabilizeLevelState(LevelState& state, RiskLevel rawLevel);

    // TTR-H 분기 한 프레임. 거리 샘플을 넣고 hState·ttrH·hRawLevel·hLevel 을 result 에 채움. P 상태는 안 건드림
    void updateHomography(TrackHistory& history, const cv::Rect& box, bool truncated, bool isLeadTarget,
                          int currentFrame, RiskResult& result) const;

    double fps_;
    int frameHeight_;
    std::size_t historySize_;
    int staleFrameLimit_;
    int maximumHistoryGapFrames_;
    float heightCollapseResetRatio_;
    bool bonnetRule_;
    bool holdRule_;
    bool bottomRule_;

    // TTR-H: setRoadHomography() 를 불렀을 때만 켜짐
    bool homographyEnabled_ = false;
    RoadHomography roadHomography_;

    std::unordered_map<int, TrackHistory> histories_;
    std::unordered_map<int, ObservationState> observationStates_;
};