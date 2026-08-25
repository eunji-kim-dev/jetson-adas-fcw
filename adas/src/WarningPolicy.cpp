#include "adas/WarningPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

/*
 * RiskAnalyzer 자체의 안정화 외에도, 상단의 큰 경고 배너는
 * 같은 선행 차량에서 위험 신호가 일정 프레임 이상 연속될 때만 표시
 *
 * 10.57초부터 장면이 바뀌기 전까지는 주의 상태가 약 5프레임만
 * 나타났으므로, 8프레임 연속 확인 조건을 적용하면
 * 이런 짧은 오경고가 상단에 뜨지 않음
 */
constexpr int cautionConfirmationFrames = 8;
constexpr int dangerConfirmationFrames = 3;

} // namespace

/*
 * warningHoldFrames: 경고가 한두 프레임 만에 사라져 깜빡이는 현상을
 * 줄이기 위한 상단 경고 유지 시간. 기존 0.4초보다 조금 줄여
 * 정상 상태로 돌아온 뒤 오래 남아 보이지 않도록 0.25초로 조정
 *
 * minimumLeadGroundYForWarning: 시간 기반 위험 조건(샘플 수, 크기,
 * 증가율, 접근 속도)은 RiskAnalyzer가 담당하고, 여기에는 화면 기하 조건만 둔다
 */
WarningPolicy::WarningPolicy(double sourceFps, int frameHeight)
    : warningHoldFrames_(std::max(1, static_cast<int>(std::round(sourceFps * 0.25)))),
      minimumLeadGroundYForWarning_(static_cast<int>(std::round(frameHeight * 0.60))) {}

void WarningPolicy::reset() {
    cautionHoldRemaining_ = 0;
    dangerHoldRemaining_ = 0;
    cautionCandidateFrames_ = 0;
    dangerCandidateFrames_ = 0;
    warningCandidateLeadId_ = -1;
}

RiskLevel WarningPolicy::bannerLevel() const {
    if (dangerHoldRemaining_ > 0) return RiskLevel::Danger;
    if (cautionHoldRemaining_ > 0) return RiskLevel::Caution;
    return RiskLevel::Safe;
}

RiskResult WarningPolicy::applyGeometryGate(const RiskResult& rawRisk, const ObjectGeometry& geometry, bool isLeadTarget) const {
    // 시간 기반 조건은 RiskAnalyzer에서 이미 판단했으므로
    // 여기서는 화면 위치와 passing-by 여부만 검사한다.
    const bool geometryAllowsWarning = isLeadTarget && (rawRisk.truncated || geometry.groundPoint.y >= minimumLeadGroundYForWarning_) && !geometry.passingBy;

    RiskResult risk = rawRisk;
    if (risk.level != RiskLevel::Safe && !geometryAllowsWarning) {
        risk.level = RiskLevel::Safe;
        risk.valid = false;
        risk.ttcSeconds = std::numeric_limits<float>::infinity();
    }
    return risk;
}

/*
 * 상단 경고 배너는 다음 세 단계를 모두 통과해야 표시
 * 1. 장면 전환 직후의 분석 대기 시간이 아닐 것
 * 2. 같은 선행 차량에서 위험 상태가 연속으로 관찰될 것
 * 3. CAUTION은 8프레임, DANGER는 3프레임 이상 지속될 것
 *
 * 장면이 바뀌면 이전 경고 유지 카운터를 즉시 0으로 만들기 때문에,
 * 새 화면에 차량이 없는데 이전 장면의 경고가 남는 현상이 사라짐
 */
void WarningPolicy::update(bool analysisEnabled, bool sceneChanged, bool leadRiskFound, int activeLeadId, RiskLevel leadLevel) {
    if (!analysisEnabled || sceneChanged) {
        reset();
    } else if (leadRiskFound && activeLeadId >= 0) {
        // 선행 차량 ID가 바뀌면 이전 차량에서 쌓인
        // 위험 연속 프레임 수를 이어받지 않음
        if (warningCandidateLeadId_ != activeLeadId) {
            warningCandidateLeadId_ = activeLeadId;
            cautionCandidateFrames_ = 0; dangerCandidateFrames_ = 0;
        }

        if (leadLevel == RiskLevel::Danger) {
            ++dangerCandidateFrames_; cautionCandidateFrames_ = 0;
            if (dangerCandidateFrames_ >= dangerConfirmationFrames) { dangerHoldRemaining_ = warningHoldFrames_; cautionHoldRemaining_ = 0; }
        } else if (leadLevel == RiskLevel::Caution) {
            ++cautionCandidateFrames_; dangerCandidateFrames_ = 0;
            if (cautionCandidateFrames_ >= cautionConfirmationFrames) cautionHoldRemaining_ = warningHoldFrames_;
            if (dangerHoldRemaining_ > 0) --dangerHoldRemaining_;
        } else {
            cautionCandidateFrames_ = 0; dangerCandidateFrames_ = 0;
            if (dangerHoldRemaining_ > 0) --dangerHoldRemaining_;
            if (cautionHoldRemaining_ > 0) --cautionHoldRemaining_;
        }
    } else {
        warningCandidateLeadId_ = -1; cautionCandidateFrames_ = 0; dangerCandidateFrames_ = 0;
        if (dangerHoldRemaining_ > 0) --dangerHoldRemaining_;
        if (cautionHoldRemaining_ > 0) --cautionHoldRemaining_;
    }
}
