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
    LeadSelector(const std::vector<cv::Point>& roadRoi, const std::vector<cv::Point>& egoLaneRoi, double sourceFps);

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

    int activeLeadId_ = -1;
    std::unordered_map<int, int> egoLaneStreakById_;
    std::unordered_map<int, int> egoLaneGraceById_;
    std::unordered_map<int, std::deque<float>> lateralHistoryById_;
    std::unordered_map<int, ObjectGeometry> geometryById_;
};
