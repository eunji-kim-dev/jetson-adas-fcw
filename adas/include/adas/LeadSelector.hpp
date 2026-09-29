#pragma once

#include "perception/Detection.hpp"

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
};

/*
 * --lead-rule 플래그 (9/28 명세). 전부 false 면 기존 규칙 그대로 (golden 보존)
 *   overlap : LEAD 후보를 접지점 lane 안 "또는" 박스–lane 가로 겹침으로 잡음 (진입 15% / 유지 10%)
 *   history : TTC-P 샘플을 lane 여부와 관계없이 차량 트랙 전부에 쌓고, 연속 프레임 높이 40% 붕괴면 이력 초기화 (RiskAnalyzer·main 에서 씀)
 *   gap     : 트랙이 3프레임까지 안 보여도 lane 체류 카운터·횡이동 이력을 지우지 않음 (증가 없이 보존)
 *   passby  : passing-by 를 바깥쪽 횡이동 "그리고" 최근 8프레임 겹침 0.1 이상 감소로 판정
 *   gate    : 60% 위치 게이트를 DANGER && 겹침 50% 이상이면 면제 (WarningPolicy 에서 씀)
 * 플래그마다 독립이라 하나씩 켜서 효과를 따로 볼 수 있음
 */
struct LeadRuleFlags {
    bool overlap = false;
    bool history = false;
    bool gap = false;
    bool passby = false;
    bool gate = false;
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
    void update(const std::vector<TrackedObject>& trackedObjects, bool analysisEnabled);

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
};
