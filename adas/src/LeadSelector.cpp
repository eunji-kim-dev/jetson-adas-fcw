#include "adas/LeadSelector.hpp"

#include "perception/Classes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {

struct LanePosition {
    bool inside = false;
    float normalizedX = 0.5F;
};

// 추월하면서 옆으로 빠지는 차량과 실제 정면 선행 차량을 구분하기 위한 횡방향 이력 길이/기울기 기준
constexpr std::size_t lateralHistorySize = 8;
constexpr float maximumLateralDriftPerFrame = 0.02F;

LanePosition calculateLanePosition(const std::vector<cv::Point>& trapezoid, const cv::Point& point) {
    if (trapezoid.size() != 4) return {};

    const float topY = (static_cast<float>(trapezoid[0].y) + static_cast<float>(trapezoid[1].y)) / 2.0F;
    const float bottomY = (static_cast<float>(trapezoid[2].y) + static_cast<float>(trapezoid[3].y)) / 2.0F;

    // 화면 하단까지 내려온 가까운 차량은 bottomY 아래라고 차선 밖으로 버리지 않음
    // verticalRatio가 아래에서 1.0으로 clamp되므로 하단 경계 기준으로 판정 가능
    if (point.y < topY || bottomY <= topY) return {};

    const float verticalRatio = std::clamp((static_cast<float>(point.y) - topY) / (bottomY - topY), 0.0F, 1.0F);
    const float leftX = static_cast<float>(trapezoid[0].x) + (static_cast<float>(trapezoid[3].x) - static_cast<float>(trapezoid[0].x)) * verticalRatio;
    const float rightX = static_cast<float>(trapezoid[1].x) + (static_cast<float>(trapezoid[2].x) - static_cast<float>(trapezoid[1].x)) * verticalRatio;

    if (point.x < leftX || point.x > rightX || rightX <= leftX) return {};
    return {true, (static_cast<float>(point.x) - leftX) / (rightX - leftX)};
}

} // namespace

// 동영상 35초 길가 트럭이 잠깐 ego lane에 들어와 바로 LEAD가 되는 문제 방지: 체류 프레임(streak) 조건
// ego lane 경계를 한두 프레임 벗어나도 기존 분석 이력을 바로 지우지 않기 위한 유예(grace)
// 컷인 차량은 정의상 ego lane에 방금 들어온 차량이므로 일반 LEAD의 0.5초 체류 조건보다 짧은 조건을 사용
LeadSelector::LeadSelector(const std::vector<cv::Point>& roadRoi, const std::vector<cv::Point>& egoLaneRoi, double sourceFps)
    : roadRoi_(roadRoi),
      egoLaneRoi_(egoLaneRoi),
      egoLaneGraceFrames_(std::max(3, static_cast<int>(std::round(sourceFps * 0.15)))),
      leadEligibilityFrames_(std::max(1, static_cast<int>(std::round(sourceFps * 0.5)))),
      cutInEligibilityFrames_(std::max(4, static_cast<int>(std::round(sourceFps * 0.15)))) {}

void LeadSelector::reset() {
    activeLeadId_ = -1;
    egoLaneStreakById_.clear();
    egoLaneGraceById_.clear();
    lateralHistoryById_.clear();
    geometryById_.clear();
}

void LeadSelector::update(const std::vector<TrackedObject>& trackedObjects, bool analysisEnabled) {
    geometryById_.clear();
    std::unordered_map<int, float> leadScoreById;
    int proposedLeadId = -1;
    float proposedLeadScore = -std::numeric_limits<float>::infinity();

    // 먼저 모든 객체의 접지점과 차선 위치를 계산하고
    // 내 차선에서 가장 가까운 선행 차량 후보를 선택
    for (const TrackedObject& trackedObject : trackedObjects) {
        const cv::Rect& box = trackedObject.box;
        const cv::Point groundPoint(box.x + box.width / 2, box.y + box.height);
        const bool insideRoad = cv::pointPolygonTest(roadRoi_, groundPoint, false) >= 0.0;
        const LanePosition lanePosition = calculateLanePosition(egoLaneRoi_, groundPoint);

        // 새 LEAD 진입 조건은 엄격하게 유지하되,
        // 이미 lane 안에 있던 차량이 경계를 잠깐 넘는 경우 이력은 유예 기간 동안 보존
        int& laneStreak = egoLaneStreakById_[trackedObject.trackId];
        int& laneGrace = egoLaneGraceById_[trackedObject.trackId];

        if (lanePosition.inside) {
            ++laneStreak;
            laneGrace = egoLaneGraceFrames_;
        } else if (laneGrace > 0) {
            --laneGrace;
        } else {
            laneStreak = 0;
        }

        const bool laneHeld = lanePosition.inside || laneGrace > 0;
        // lane 중앙에서 바깥쪽으로 이동하면 passing-by,
        // 반대로 중앙으로 빠르게 접근하면 cutting-in으로 판단
        auto& lateralHistory = lateralHistoryById_[trackedObject.trackId];

        if (lanePosition.inside) {
            lateralHistory.push_back(lanePosition.normalizedX);
            if (lateralHistory.size() > lateralHistorySize) lateralHistory.pop_front();
        } else if (!laneHeld) {
            lateralHistory.clear();
        }

        float outwardDriftPerFrame = 0.0F;
        if (lateralHistory.size() >= 4) {
            const float pastOffset = std::abs(lateralHistory.front() - 0.5F);
            const float recentOffset = std::abs(lateralHistory.back() - 0.5F);
            outwardDriftPerFrame = (recentOffset - pastOffset) / static_cast<float>(lateralHistory.size() - 1);
        }

        const bool passingBy = outwardDriftPerFrame > maximumLateralDriftPerFrame;
        const bool cuttingIn = outwardDriftPerFrame < -maximumLateralDriftPerFrame;
        const int requiredStreak = cuttingIn ? cutInEligibilityFrames_ : leadEligibilityFrames_;
        float leadScore = -std::numeric_limits<float>::infinity();

        if (lanePosition.inside && isVehicleClass(trackedObject.classId) && laneStreak >= requiredStreak) {
            const float centerPenalty = std::abs(lanePosition.normalizedX - 0.5F) * 90.0F;
            leadScore = static_cast<float>(groundPoint.y) - centerPenalty;
            leadScoreById[trackedObject.trackId] = leadScore;

            if (leadScore > proposedLeadScore) {
                proposedLeadScore = leadScore;
                proposedLeadId = trackedObject.trackId;
            }
        }
        geometryById_[trackedObject.trackId] = {groundPoint, insideRoad, lanePosition.inside, laneHeld, lanePosition.normalizedX, leadScore, passingBy};
    }

    // 이번 프레임에 보이지 않는 Track의 lane 체류/횡이동 이력은 제거
    for (auto iterator = egoLaneStreakById_.begin(); iterator != egoLaneStreakById_.end();) {
        if (geometryById_.find(iterator->first) == geometryById_.end()) iterator = egoLaneStreakById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = egoLaneGraceById_.begin(); iterator != egoLaneGraceById_.end();) {
        if (geometryById_.find(iterator->first) == geometryById_.end()) iterator = egoLaneGraceById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = lateralHistoryById_.begin(); iterator != lateralHistoryById_.end();) {
        if (geometryById_.find(iterator->first) == geometryById_.end()) iterator = lateralHistoryById_.erase(iterator);
        else ++iterator;
    }

    const auto activeLeadIterator = leadScoreById.find(activeLeadId_);
    const bool activeLeadVisible = activeLeadIterator != leadScoreById.end();

    // 장면 전환 직후에는 선행 차량을 바로 선택하지 않음
    // 새 장면에서 추적 정보가 다시 안정화된 뒤에만 선택
    if (!analysisEnabled) activeLeadId_ = -1;
    else if (!activeLeadVisible) activeLeadId_ = proposedLeadId;
    else if (proposedLeadId >= 0 && proposedLeadId != activeLeadId_ && proposedLeadScore > activeLeadIterator->second + 35.0F) activeLeadId_ = proposedLeadId;
}
