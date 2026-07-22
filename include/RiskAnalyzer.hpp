#pragma once

// TrackedObject (trackId, classId, confidence, box) 사용
#include "Detection.hpp"

#include <cstddef>
#include <deque>       // ID별 최근 프레임 기록을 밀어넣고 빼내기 편해서 사용
#include <limits>      // TTC 계산 불가할 때 무한대로 표시하려고
#include <string>
#include <unordered_map> // trackId -> TrackHistory

// 접근 위험 단계. 숫자가 클수록 위험함
enum class RiskLevel {
    Safe = 0,     // 위험 없음 or 아직 판단할 데이터가 부족함
    Caution = 1,  // 앞차와 간격이 줄고 있음, 주의 필요
    Danger = 2    // 빠르게 접근 중, 즉각 경고 필요
};

// RiskAnalyzer::update() 한 번 호출할 때마다 나오는 결과값
struct RiskResult {
    RiskLevel level = RiskLevel::Safe; // 여러 프레임 검증 거쳐서 안정화된 최종 단계

    // false면 아래 수치들은 신뢰하면 안 됨:
    // - 애초에 lead 대상이 아니었거나
    // - 기록이 아직 부족하거나
    // - 박스가 커지고 있지 않은 경우 등
    bool valid = false;

    // 박스 높이 변화 기반 TTC 근사치 (실제 미터 단위 거리 계산 아님, 순전히 화면상 크기 변화 기반)
    // 접근 안 하거나 계산 불가하면 무한대로 둠
    float ttcSeconds = std::numeric_limits<float>::infinity();

    // log(박스 높이)의 초당 변화율. TTC 계산의 핵심 값.
    // 양수=가까워지는 중, 0 근처=변화 없음, 음수=멀어지는 중
    float logHeightRatePerSecond = 0.0F;

    // 박스 하단 중앙점이 초당 몇 픽셀 움직였는지 (양수=아래로, 즉 화면상 접근 방향)
    // 박스 높이 증가랑 같이 봐서 진짜 접근인지 확인하는 보조 지표
    float groundSpeedPixelsPerSecond = 0.0F;

    // 과거 평균 높이 대비 최근 평균 높이 비율. 1.10이면 최근에 10% 커졌다는 뜻
    float heightGrowthRatio = 1.0F;

    // 이 트랙에 대해 지금까지 쌓인 샘플 개수 (디버깅/화면 표시용)
    int sampleCount = 0;
};

// trackId별로 박스 높이 변화를 추적해서 TTC 근사치랑 위험 단계를 뽑아내는 클래스
class RiskAnalyzer {
public:
    // fps: 프레임 번호 차이를 초 단위로 환산할 때 필요
    // historySize: ID 하나당 들고 있을 최근 샘플 개수 (기본 15프레임, 30fps면 약 0.5초)
    // staleFrameLimit: 화면에서 안 보인 지 이만큼 지나면 기록 삭제 (기본 60프레임 = 30fps 기준 2초)
    explicit RiskAnalyzer(
        double fps,
        int historySize = 15,
        int staleFrameLimit = 60
    );

    // 이번 프레임 트랙 하나를 분석함
    // isLeadTarget이 false면 (선행 차량이 아니면) 기록 리셋하고 그냥 Safe 반환
    // currentFrame은 프레임 간격 계산 + 공백 확인용
    RiskResult update(
        const TrackedObject& trackedObject,
        bool isLeadTarget,
        int currentFrame
    );

    // 오래 안 보인 트랙 기록 정리 (안 지우면 histories_가 계속 쌓임)
    void removeStaleTracks(
        int currentFrame
    );

    // RiskLevel -> 화면 표시용 문자열 ("SAFE" / "CAUTION" / "DANGER")
    static std::string toString(
        RiskLevel level
    );

private:
    // 한 프레임 시점의 원시 관측값
    struct Sample {
        int frameIndex;   // 정확한 시간 간격 계산용
        float boxHeight;  // 가까워질수록 커짐
        float groundY;    // 접지점 y좌표, 화면 아래로 내려가는지 볼 때 씀
    };

    // trackId 하나에 대해 유지하는 위험 분석 상태 전체
    struct TrackHistory {
        std::deque<Sample> samples; // historySize_ 넘으면 오래된 것부터 버림

        int lastSeenFrame = -1; // -1이면 아직 한 번도 안 잡힘

        RiskLevel stableLevel = RiskLevel::Safe;   // 실제로 화면에 표시되는 확정 단계
        RiskLevel pendingLevel = RiskLevel::Safe;  // 지금 막 올라오려는 중인 단계

        int pendingFrames = 0;    // pendingLevel이 몇 프레임째 연속인지
                                  // -> 일정 프레임 이상 버텨야 stableLevel로 승격
        int lowerLevelFrames = 0; // stableLevel보다 낮은 값이 몇 프레임째 연속인지
                                  // -> 한 프레임 반짝 좋아졌다고 바로 SAFE로 안 내려가게
    };

    // 샘플들 갖고 시간 대비 변화율 선형회귀로 뽑아냄
    // useLogHeight=true면 log(높이) 기울기, false면 groundY 기울기
    static float calculateRegressionSlope(
        const std::deque<Sample>& samples,
        double fps,
        bool useLogHeight
    );

    // samples[start .. start+count) 구간 박스 높이 평균
    // (과거 구간 vs 최근 구간 평균 비교해서 heightGrowthRatio 낼 때 씀)
    static float calculateAverageHeight(
        const std::deque<Sample>& samples,
        std::size_t start,
        std::size_t count
    );

    // 트랙 기록 초기화. lead 대상에서 빠지거나 공백이 너무 길어지면 호출됨
    // (예전 기록이랑 지금 기록을 이어붙여서 분석하면 안 되니까)
    static void resetTrackHistory(
        TrackHistory& history
    );

    // 이번 프레임 rawLevel을 바로 반영하지 않고 시간적으로 안정화시킴
    // 올라갈 때: 같은 단계가 몇 프레임 이어져야 반영
    // 내려갈 때: 낮은 단계가 일정 시간 유지돼야 반영
    // -> 경고가 프레임마다 깜빡이는 거 방지
    RiskLevel stabilizeLevel(
        TrackHistory& history,
        RiskLevel rawLevel
    ) const;

    double fps_;

    std::size_t historySize_;

    int staleFrameLimit_;

    // 같은 ID가 다시 나타나도 공백이 이 값보다 크면 이전 기록이랑 이어서 안 씀
    // (한참 놓쳤다가 다시 잡힌 객체로 TTC 이상하게 계산되는 거 막으려고)
    int maximumHistoryGapFrames_;

    // trackId -> TrackHistory
    std::unordered_map<
        int,
        TrackHistory
    > histories_;
};