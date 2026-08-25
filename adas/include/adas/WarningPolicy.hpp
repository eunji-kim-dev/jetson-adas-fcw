#pragma once

#include "adas/LeadSelector.hpp"
#include "adas/RiskAnalyzer.hpp"

/*
 * 상단 경고 배너 표시 정책
 *
 * RiskAnalyzer가 판단한 시간 기반 위험도에 두 가지를 추가로 적용한다.
 *   1. 화면 기하 게이트 — 선행 차량이 화면 하단에 충분히 내려와 있고
 *      추월로 빠지는 중이 아닐 때만 위험 단계를 인정
 *   2. 연속 확인/유지 — 같은 선행 차량에서 위험 신호가 일정 프레임
 *      이상 연속될 때만 배너를 띄우고, 정상 복귀 후 짧게 유지
 *
 * 위험도 판단 자체(TTC-P, 성장률, 접근 속도)는 RiskAnalyzer의 몫이고
 * 이 클래스는 "언제 사용자에게 보여줄 것인가"만 담당한다.
 */
class WarningPolicy {
public:
    WarningPolicy(double sourceFps, int frameHeight);

    // 화면 기하 조건을 통과하지 못한 위험 단계를 Safe로 되돌린다
    RiskResult applyGeometryGate(const RiskResult& rawRisk, const ObjectGeometry& geometry, bool isLeadTarget) const;

    // 한 프레임의 LEAD 위험 상태로 배너 연속 확인/유지 카운터를 갱신
    void update(bool analysisEnabled, bool sceneChanged, bool leadRiskFound, int activeLeadId, RiskLevel leadLevel);

    // 장면 전환 시 이전 장면의 경고 상태를 즉시 제거
    void reset();

    // 현재 표시할 배너 단계 (Safe면 표시하지 않음)
    RiskLevel bannerLevel() const;

private:
    const int warningHoldFrames_;
    const int minimumLeadGroundYForWarning_;

    int cautionHoldRemaining_ = 0;
    int dangerHoldRemaining_ = 0;
    int cautionCandidateFrames_ = 0;
    int dangerCandidateFrames_ = 0;
    int warningCandidateLeadId_ = -1;
};
