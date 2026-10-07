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
 *   edge    : 박스가 화면 좌우 끝(2px 안)에 닿았고 접지점이 lane 밖이면, 겹침만으로 "새로" 후보가 되지 못함
 *             화면 끝에 닿음 = 차 일부가 화면 밖 = 옆에 나란히 있는 차임. 박스 아래 변(가까운 옆구리)과 안쪽 변(먼 뒤 모서리)이
 *             서로 다른 깊이에서 나와 겹침이 실제보다 크게 잡힘 (1_009 옆 SUV·6_014·06 옆 차가 lane 안 앞차를 이겼음)
 *             이미 후보인 트랙(체류 > 0)은 막지 않음. 화면 안에서 들어온 컷인 차가 충돌 직전 화면 끝에 닿아도 유지됨 (11: 72)
 *             한 번 이 조건으로 화면 끝에 닿은 트랙은 "옆 차"로 기억함. 기억 중에는 이미 후보였어도 접지점이 lane 밖인 프레임은 후보가 못 됨
 *             (접지점이 lane 안인 프레임만 후보가 되고, 그래도 기억은 안 지움). 기억은 박스가 화면 좌우 끝에서 20px 넘게 떨어진
 *             영상 프레임이 3개 연속일 때만 풀림. 안 보인 프레임(gap)·held 프레임이 끼면 0 부터 다시 셈
 *             (17_072: 버스 박스 오른쪽 끝이 1917~1919 로 흔들려 124 한 프레임만 "안 닿음"이 됐고 그 프레임에 후보가 된 뒤 128 에 LEAD 가 됐음.
 *              기억만 두고 후보가 되면 지우면, 198 처럼 접지점이 한 프레임만 lane 안에 찍혀도 후보 → 199·200 은 "이미 후보" 예외로 이어짐)
 *             남은 구멍: 화면 끝에 잘린 박스는 박스 가운데가 실제 차보다 화면 안쪽에 찍혀 접지점이 lane 안으로 들어올 수 있고, 그 프레임은 후보가 됨
 *             (화면 끝에 닿은 채 가까이 끼어드는 차를 막지 않으려고 그대로 둠. 잘린 박스의 접지점 처리는 따로 정할 일)
 *             접지점이 lane 안인 차는 해당 없음. 검은 테두리가 있는 영상은 테두리가 화면 끝이 아니라서 안 걸림
 *   bottom  : 박스가 화면 안에서 아래 변만 놓친 프레임(높이가 직전 정상 프레임보다 줄고 폭÷높이는 15% 넘게 커짐)을 화면 아래 절단과 같이
 *             폭 ÷ 종횡비로 높이를 보정해 TTC-P 이력을 이음. 5프레임 넘게 이어지면 새 모양으로 받아들임 (RiskAnalyzer 에서 씀)
 *             (07 86~89 에서 트럭 아래 변이 올라가 이력이 오염되고, 돌아온 90 이 급변으로 잡혀 93 에 리셋 → 충돌 102 까지 배너 못 띄움)
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
    bool edge = false;
    bool bottom = false;
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
    // frameWidth: edge 규칙이 박스가 화면 좌우 끝에 닿았는지 볼 때 씀. 0 이면 edge 를 켜도 아무것도 안 함
    LeadSelector(const std::vector<cv::Point>& roadRoi, const std::vector<cv::Point>& egoLaneRoi, double sourceFps,
                 const LeadRuleFlags& rules = LeadRuleFlags(), int frameWidth = 0);

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
    const int frameWidth_;   // edge: 화면 폭. 0 이면 edge 를 안 씀

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
    // edge: "옆 차"로 기억한 트랙. 값은 박스가 화면 좌우 끝에서 확실히 떨어진 연속 영상 프레임 수 (풀림 판정용)
    std::unordered_map<int, int> edgeSideClearFramesById_;
};
