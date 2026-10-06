#pragma once

#include "perception/Detection.hpp"
#include "adas/RiskAnalyzer.hpp"

#include <opencv2/core.hpp>
#include <deque>
#include <unordered_map>
#include <vector>

struct ObjectGeometry {
    cv::Point groundPoint;
    bool insideRoad = false;
    bool insideEgoLane = false;

    // ego lane 경계를 순간적으로 벗어나도 위험 분석 이력은 잠시 유지
    // 새 LEAD 후보 선정에는 쓰지 않고 기존 LEAD/분석 이력 보존에만 사용
    bool laneHeld = false;

    float normalizedLaneX = 0.5F;
    float leadScore = -std::numeric_limits<float>::infinity();
    bool passingBy = false;

    // 박스 가로 폭 중 ego lane 안에 든 비율 (0~1). --lead-rule 의 overlap/passby/gate 중 하나라도 켰을 때만 계산되고
    // 아니면 0 으로 남음. WarningPolicy 의 gate 면제 판단에 씀
    float laneOverlap = 0.0F;

    // bonnet: 보닛 반사 확정 트랙. 후보·분석 대상에서 빠지고 라벨 표시에만 씀
    bool bonnetSuspect = false;
    // hold/bonnet: 이번 프레임 관측이 비정상이라 마지막 정상 프레임의 기하를 재사용한 상태
    bool held = false;

    // rank: 겹침만으로 들어온 후보에 준 감점 (px). 접지점 lane 안 후보·규칙 꺼짐이면 0. 선정 로그용
    float rankPenalty = 0.0F;

    // ★ 선정 로그용. leadScore 가 비어 있을 때 "후보 밖"인지 "체류 부족"인지 가르기 위해 둠
    bool laneCandidate = false;   // 이번 프레임 후보 조건(접지점 lane 안 또는 overlap 겹침) 통과
    int laneStreak = 0;           // 후보 연속 체류 프레임 (gap·held 면 동결값)
    int requiredStreak = 0;       // LEAD 자격에 필요한 체류 (컷인이면 짧음)
    bool eligible = false;        // laneCandidate && 차량 클래스 && laneStreak >= requiredStreak
};

/*
 * --lead-rule 플래그 (9/28 명세). 전부 false 면 기존 규칙 그대로 (golden 보존)
 *   overlap : LEAD 후보를 접지점 lane 안 "또는" 박스–lane 가로 겹침으로 잡음 (진입 15% / 유지 10%)
 *   history : TTC-P 샘플을 lane 여부와 관계없이 차량 트랙 전부에 쌓고, 연속 프레임 높이 40% 붕괴면 이력 초기화 (RiskAnalyzer·main 에서 씀)
 *   gap     : 트랙이 3프레임까지 안 보여도 lane 체류 카운터·횡이동 이력을 지우지 않음 (증가 없이 보존)
 *   passby  : passing-by 를 바깥쪽 횡이동 "그리고" 최근 8프레임 겹침 0.1 이상 감소로 판정
 *   gate    : 60% 위치 게이트를 DANGER && 겹침 50% 이상이면 면제 (WarningPolicy 에서 씀)
 *   bonnet  : 보닛 반사 모양 박스(아래 변 ≥ 95%H, 위 변 ≥ 55%H, 폭÷높이 ≥ 2.5)를 첫 프레임부터 비정상 관측으로 보고
 *             3프레임 넘게 이어지면 확정해 후보·분석에서 뺌. 정상 모양이 나오면 해제 (RiskAnalyzer 가 판정, 여기서는 후보 제외)
 *   hold    : history 의 높이 급변을 리셋 대신 3프레임까지 보존. 보존 프레임은 샘플·체류·횡이동·겹침 이력을 동결하고
 *             기존 LEAD 는 유지하되 새 후보로는 안 냄. 배너 카운터도 동결 (WarningPolicy). history 없이는 아무것도 안 함
 *   rank    : 경쟁 후보 사이의 우선순위. 접지점이 lane 밖이고 겹침으로만 들어온 후보는 점수에서 (1 − 겹침비율) × 70 을 뺌
 *             후보 진입은 그대로라 그 후보가 유일하면 여전히 LEAD 가 됨 (07 옆 밴·13 옆 버스가 접지점 y 만으로 lane 안 앞차를 이겼음)
 */

struct LeadRuleFlags {
    bool overlap = false;
    bool history = false;
    bool gap = false;
    bool passby = false;
    bool gate = false;
    bool bonnet = false;
    bool hold = false;
    bool rank = false;
};

/*
 * 내 차선 선행 차량(LEAD) 선택기
 *
 * TrackedObject 목록을 받아 각 객체의 접지점/차선 위치를 계산하고,
 * ego lane 체류 시간·횡방향 이동·컷인 여부를 종합해
 * 위험 분석 대상이 될 선행 차량 한 대를 히스테리시스와 함께 유지
 *
 * ROI(도로/내 차선 사다리꼴)는 카메라 장착 기하에 따라 달라지므로
 * 앱이 설정으로 주입한다
 */
class LeadSelector {
public:
    // rules 를 생략하면 전부 꺼진 기존 규칙임
    LeadSelector(const std::vector<cv::Point>& roadRoi, const std::vector<cv::Point>& egoLaneRoi, double sourceFps,
                 const LeadRuleFlags& rules = LeadRuleFlags());

    // 한 프레임의 추적 결과로 기하 정보와 LEAD 선택을 갱신
    // analysisEnabled=false(장면 전환 워밍업)면 LEAD를 선택하지 않음
    // observations: RiskAnalyzer::classifyObservations() 의 이번 프레임 판정. nullptr 이거나 비어 있으면 기존 동작
    void update(const std::vector<TrackedObject>& trackedObjects, bool analysisEnabled,
                const std::unordered_map<int, ObservationState>* observations = nullptr);

    // 장면 전환 시 lane 체류/횡이동 이력과 LEAD 선택을 초기화
    void reset();

    int activeLeadId() const { return activeLeadId_; }
    const std::unordered_map<int, ObjectGeometry>& geometryById() const { return geometryById_; }

private:
    std::vector<cv::Point> roadRoi_;
    std::vector<cv::Point> egoLaneRoi_;

    const int egoLaneGraceFrames_;
    const int leadEligibilityFrames_;
    const int cutInEligibilityFrames_;
    const LeadRuleFlags rules_;

    int activeLeadId_ = -1;
    std::unordered_map<int, int> egoLaneStreakById_;
    std::unordered_map<int, int> egoLaneGraceById_;
    // 횡이동 이력 한 칸. frame 은 update() 호출 횟수 기준 — gap 규칙에서 미관측 프레임을 건너뛴 이력의 실제 간격을 알기 위해 같이 둠
    struct LateralSample {
        int frame;
        float normalizedX;
    };
    // passby: 프레임별 박스–lane 겹침 비율 한 칸 (최근 8프레임 창)
    struct OverlapSample {
        int frame;
        float ratio;
    };

    int frameIndex_ = 0;   // update() 가 불릴 때마다 1 증가. 이력의 frame 값 기준
    std::unordered_map<int, std::deque<LateralSample>> lateralHistoryById_;
    std::unordered_map<int, std::deque<OverlapSample>> overlapHistoryById_;
    // gap: 트랙이 연속으로 안 보인 프레임 수. 3 을 넘으면 위 이력을 지움
    std::unordered_map<int, int> unseenFramesById_;
    std::unordered_map<int, ObjectGeometry> geometryById_;
    // hold/bonnet: 트랙별 마지막 정상 프레임의 기하. held 프레임에 재사용함
    std::unordered_map<int, ObjectGeometry> lastGeometryById_;
};
