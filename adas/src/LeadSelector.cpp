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

// --lead-rule 기준값 (9/28, 영상 10·11·01·08 로그로 잡음. 음성 영상으로 과적합을 볼 때까지 고정)
constexpr float overlapEntryRatio = 0.15F;       // overlap: 이 겹침 이상이면 lane 밖 접지점이어도 후보 진입
constexpr float overlapKeepRatio = 0.10F;        // overlap: 이미 후보면 이 겹침까지는 후보 유지
constexpr int gapPreserveFrames = 3;             // gap: 이 프레임 수까지 미관측이어도 이력 보존
constexpr std::size_t overlapHistorySize = 8;    // passby: 겹침 추세를 보는 창
constexpr float passByOverlapDrop = 0.10F;       // passby: 창 안에서 겹침이 이만큼 줄어야 passing-by

struct LaneOverlap {
    bool valid = false;         // 박스 아래 변이 lane 사다리꼴 세로 범위 안에 있음
    float ratio = 0.0F;         // 박스 가로 폭 중 lane 안에 든 비율 (0~1)
    float normalizedX = 0.5F;   // 접지점의 lane 안 위치. lane 밖이면 0 미만·1 초과 (클램프 안 함)
};

// 박스 아래 변 높이에서 lane 좌우 경계를 구해 박스–lane 가로 겹침을 계산함
// 컷인·근접 차량은 원근 때문에 접지점이 lane 바깥으로 밀리면서도 박스는 lane 과 겹치며 커짐
// 접지점 하나로 후보를 잡으면 이 구간을 전부 버리므로 (10: 22프레임, 11: 29프레임) 겹침으로 후보를 잡음
// 좌우 경계 계산은 calculateLanePosition 과 같은 식임
LaneOverlap calculateLaneOverlap(const std::vector<cv::Point>& trapezoid, const cv::Rect& box, const cv::Point& groundPoint) {
    if (trapezoid.size() != 4) return {};

    const float topY = (static_cast<float>(trapezoid[0].y) + static_cast<float>(trapezoid[1].y)) / 2.0F;
    const float bottomY = (static_cast<float>(trapezoid[2].y) + static_cast<float>(trapezoid[3].y)) / 2.0F;
    if (groundPoint.y < topY || bottomY <= topY) return {};

    const float verticalRatio = std::clamp((static_cast<float>(groundPoint.y) - topY) / (bottomY - topY), 0.0F, 1.0F);
    const float leftX = static_cast<float>(trapezoid[0].x) + (static_cast<float>(trapezoid[3].x) - static_cast<float>(trapezoid[0].x)) * verticalRatio;
    const float rightX = static_cast<float>(trapezoid[1].x) + (static_cast<float>(trapezoid[2].x) - static_cast<float>(trapezoid[1].x)) * verticalRatio;
    if (rightX <= leftX || box.width <= 0) return {};

    LaneOverlap result;
    result.valid = true;
    result.normalizedX = (static_cast<float>(groundPoint.x) - leftX) / (rightX - leftX);

    const float boxLeft = static_cast<float>(box.x);
    const float boxRight = static_cast<float>(box.x + box.width);
    const float intersection = std::max(0.0F, std::min(boxRight, rightX) - std::max(boxLeft, leftX));
    result.ratio = intersection / static_cast<float>(box.width);
    return result;
}

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
LeadSelector::LeadSelector(const std::vector<cv::Point>& roadRoi, const std::vector<cv::Point>& egoLaneRoi, double sourceFps,
                           const LeadRuleFlags& rules)
    : roadRoi_(roadRoi),
      egoLaneRoi_(egoLaneRoi),
      egoLaneGraceFrames_(std::max(3, static_cast<int>(std::round(sourceFps * 0.15)))),
      leadEligibilityFrames_(std::max(1, static_cast<int>(std::round(sourceFps * 0.5)))),
      cutInEligibilityFrames_(std::max(4, static_cast<int>(std::round(sourceFps * 0.15)))),
      rules_(rules) {}

void LeadSelector::reset() {
    activeLeadId_ = -1;
    egoLaneStreakById_.clear();
    egoLaneGraceById_.clear();
    lateralHistoryById_.clear();
    overlapHistoryById_.clear();
    unseenFramesById_.clear();
    geometryById_.clear();
    lastGeometryById_.clear();
}

void LeadSelector::update(const std::vector<TrackedObject>& trackedObjects, bool analysisEnabled,
                          const std::unordered_map<int, ObservationState>* observations) {
    // 이력의 frame 값. 절대 프레임 번호가 아니라 호출 횟수지만 간격 계산에는 충분함
    ++frameIndex_;
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

        // 겹침은 overlap·passby·gate 중 하나라도 켰을 때만 계산함 (기본 규칙은 접지점만 씀)
        const bool needOverlap = rules_.overlap || rules_.passby || rules_.gate;
        const LaneOverlap laneOverlap = needOverlap ? calculateLaneOverlap(egoLaneRoi_, box, groundPoint) : LaneOverlap{};

        // hold/bonnet: RiskAnalyzer 의 이번 프레임 관측 판정
        const ObservationState* observation = nullptr;
        if (observations != nullptr) {
            const auto observationIterator = observations->find(trackedObject.trackId);
            if (observationIterator != observations->end()) observation = &observationIterator->second;
        }
        if (observation != nullptr && observation->bonnet) {
            // 보닛 확정: 후보 불가. 쌓여 있던 체류·유예·횡이동·겹침 이력도 지움
            // (실제 차 ID 가 보닛 박스로 옮겨 붙은 경우 옛 streak 로 바로 LEAD 가 되지 않게)
            egoLaneStreakById_.erase(trackedObject.trackId);
            egoLaneGraceById_.erase(trackedObject.trackId);
            lateralHistoryById_.erase(trackedObject.trackId);
            overlapHistoryById_.erase(trackedObject.trackId);
            unseenFramesById_.erase(trackedObject.trackId);
            lastGeometryById_.erase(trackedObject.trackId);
            ObjectGeometry geometry;
            geometry.groundPoint = groundPoint;
            geometry.insideRoad = insideRoad;
            geometry.bonnetSuspect = true;
            geometryById_[trackedObject.trackId] = geometry;
            continue;
        }
        if (observation != nullptr && observation->held) {
            // 보존: 체류·유예·횡이동·겹침·미관측 이력을 전부 건드리지 않고 마지막 정상 프레임의 기하를 재사용함
            // 기존 LEAD 는 점수를 그대로 등록해 유지하고, 새 LEAD 후보로는 내지 않음 (proposedLead 갱신 없음)
            const auto previousGeometry = lastGeometryById_.find(trackedObject.trackId);
            if (previousGeometry != lastGeometryById_.end()) {
                ObjectGeometry geometry = previousGeometry->second;
                geometry.held = true;
                geometryById_[trackedObject.trackId] = geometry;
                if (trackedObject.trackId == activeLeadId_ && std::isfinite(geometry.leadScore)) {
                    leadScoreById[trackedObject.trackId] = geometry.leadScore;
                }
                continue;
            }
            // 정상 프레임이 한 번도 없던 트랙(보닛 모양으로 태어난 트랙 등)은 기하만 기록하고 후보로 안 봄
            ObjectGeometry geometry;
            geometry.groundPoint = groundPoint;
            geometry.insideRoad = insideRoad;
            geometry.held = true;
            geometryById_[trackedObject.trackId] = geometry;
            continue;
        }

        // 새 LEAD 진입 조건은 엄격하게 유지하되,
        // 이미 lane 안에 있던 차량이 경계를 잠깐 넘는 경우 이력은 유예 기간 동안 보존
        int& laneStreak = egoLaneStreakById_[trackedObject.trackId];
        int& laneGrace = egoLaneGraceById_[trackedObject.trackId];

        // gap: 이번 프레임에 보였으니 미관측 카운터를 0으로
        if (rules_.gap) unseenFramesById_[trackedObject.trackId] = 0;

        // overlap: 후보 = 접지점 lane 안 "또는" 겹침 ≥ 진입 15%
        // 이미 후보였던 트랙(streak > 0)은 유지 10% 까지 후보로 봄. 유지에 못 미치면 아래 기존 유예(3프레임, streak 동결)를 거쳐 이탈함
        // 높이 조건은 두지 않음 — 크기 증가는 접근의 증거지 경로 공유의 증거가 아님 (옆 차 추월도 커짐)
        bool laneCandidate = lanePosition.inside;
        if (rules_.overlap && !laneCandidate && laneOverlap.valid) {
            laneCandidate = laneOverlap.ratio >= (laneStreak > 0 ? overlapKeepRatio : overlapEntryRatio);
        }

        // 유예 판정에 쓸 "이번 프레임에 들어올 때의" 유예 잔량. 아래에서 줄이기 전 값임
        const int graceBeforeUpdate = laneGrace;

        if (laneCandidate) {
            ++laneStreak;
            laneGrace = egoLaneGraceFrames_;
        } else if (laneGrace > 0) {
            --laneGrace;
        } else {
            laneStreak = 0;
        }

        // 기존 규칙은 줄인 뒤 값으로 판정해서 유예 마지막 프레임(잔량 1→0)에 streak 는 남는데 laneHeld 는 false 가 됨
        // (그 프레임에 횡이동 이력이 지워지고, history 규칙이 없으면 TTC 이력도 끊김)
        // overlap: 줄이기 전 값으로 판정해 streak 동결 3프레임과 laneHeld 3프레임을 같은 시점에 끝냄. 기존 경로는 그대로 둠 (golden)
        const bool laneHeld = laneCandidate || (rules_.overlap ? graceBeforeUpdate > 0 : laneGrace > 0);
        // lane 중앙에서 바깥쪽으로 이동하면 passing-by,
        // 반대로 중앙으로 빠르게 접근하면 cutting-in으로 판단
        auto& lateralHistory = lateralHistoryById_[trackedObject.trackId];
        // 횡이동 창·속도를 프레임 번호 기준으로 계산할지. LeadSelector 쪽 플래그가 하나라도 켜지면 켬 (OFF 경로만 옛 계산 유지)
        const bool frameBasedLateral = rules_.overlap || rules_.gap || rules_.passby;

        // overlap 으로 잡힌 lane 밖 후보는 클램프 없는 normX 를 저장함
        // 0~1 로 클램프하면 경계에 붙은 값만 남아 바깥쪽 이동(이탈)이 이력에서 사라짐
        if (laneCandidate) {
            lateralHistory.push_back({frameIndex_, lanePosition.inside ? lanePosition.normalizedX : laneOverlap.normalizedX});
            if (lateralHistory.size() > lateralHistorySize) lateralHistory.pop_front();
            // 새 규칙 경로: "최근 8프레임" 창을 관측 개수가 아니라 프레임 차이로 자름
            // 이력은 후보일 때만 쌓여서 유예 구간(관측은 되지만 후보 아님)과 gap 의 미관측 구간이 끼면 개수 8 ≠ 8프레임임
            if (frameBasedLateral) {
                while (!lateralHistory.empty() && frameIndex_ - lateralHistory.front().frame >= static_cast<int>(lateralHistorySize)) lateralHistory.pop_front();
            }
        } else if (!laneHeld) {
            lateralHistory.clear();
        }

        // passby: 겹침 추세 창은 lane 여부와 관계없이 매 관측 프레임 기록하고, 창은 프레임 차이로 자름
        // (gap 없이는 미관측 트랙의 이력이 바로 지워져 관측이 연속이므로 개수로 자르는 것과 같음)
        if (rules_.passby) {
            auto& overlapHistory = overlapHistoryById_[trackedObject.trackId];
            overlapHistory.push_back({frameIndex_, laneOverlap.ratio});
            while (!overlapHistory.empty() && frameIndex_ - overlapHistory.front().frame >= static_cast<int>(overlapHistorySize)) overlapHistory.pop_front();
        }

        float outwardDriftPerFrame = 0.0F;
        if (lateralHistory.size() >= 4) {
            const float pastOffset = std::abs(lateralHistory.front().normalizedX - 0.5F);
            const float recentOffset = std::abs(lateralHistory.back().normalizedX - 0.5F);
            // 기존 규칙은 관측 개수 - 1 로 나눔. 새 규칙 경로는 이력에 빠진 프레임(유예·미관측)이 끼어 있으므로 실제 프레임 차이로 나눔
            // (개수로 나누면 유예 3프레임 뒤 복귀한 트랙의 횡이동 속도가 1.4~1.75배로 계산돼 passing-by 가 잘못 붙음)
            const int frameSpan = frameBasedLateral
                ? lateralHistory.back().frame - lateralHistory.front().frame
                : static_cast<int>(lateralHistory.size()) - 1;
            outwardDriftPerFrame = (recentOffset - pastOffset) / static_cast<float>(std::max(frameSpan, 1));
        }

        bool passingBy = outwardDriftPerFrame > maximumLateralDriftPerFrame;
        if (rules_.passby) {
            // passby: 충돌하는 컷인 차량도 접지점은 바깥쪽으로 밀리므로 (10: normX 0.18→0.07)
            // 바깥쪽 횡이동만으로는 추월과 못 가름. 겹침이 줄고 있을 때만 passing-by 로 봄. 높이는 안 봄
            const auto& overlapHistory = overlapHistoryById_[trackedObject.trackId];
            const bool overlapDecreasing = overlapHistory.size() >= 2 && (overlapHistory.front().ratio - overlapHistory.back().ratio) >= passByOverlapDrop;
            passingBy = passingBy && overlapDecreasing;
        }
        const bool cuttingIn = outwardDriftPerFrame < -maximumLateralDriftPerFrame;
        const int requiredStreak = cuttingIn ? cutInEligibilityFrames_ : leadEligibilityFrames_;
        float leadScore = -std::numeric_limits<float>::infinity();

        if (laneCandidate && isVehicleClass(trackedObject.classId) && laneStreak >= requiredStreak) {
            // 중앙 패널티 계산 때만 클램프함. lane 밖 후보는 경계값(0 또는 1)으로 계산돼 lane 안 차량보다 점수가 낮음
            const float normalizedX = lanePosition.inside ? lanePosition.normalizedX : std::clamp(laneOverlap.normalizedX, 0.0F, 1.0F);
            const float centerPenalty = std::abs(normalizedX - 0.5F) * 90.0F;
            leadScore = static_cast<float>(groundPoint.y) - centerPenalty;
            leadScoreById[trackedObject.trackId] = leadScore;

            if (leadScore > proposedLeadScore) {
                proposedLeadScore = leadScore;
                proposedLeadId = trackedObject.trackId;
            }
        }
        geometryById_[trackedObject.trackId] = {groundPoint, insideRoad, lanePosition.inside, laneHeld, lanePosition.normalizedX, leadScore, passingBy, laneOverlap.ratio};
        // hold/bonnet: 정상 관측 프레임의 기하를 다음 held 프레임용으로 남김
        lastGeometryById_[trackedObject.trackId] = geometryById_[trackedObject.trackId];
    }

    // 이번 프레임에 보이지 않는 Track의 lane 체류/횡이동 이력은 제거
    // gap: 3프레임까지는 지우지 않고 그대로 둠 (streak·grace 증가 없이 동결). 그 뒤에 지움
    // 11번은 67·68·71 에서 트랙이 한 프레임씩 끊겨 체류 카운터가 0 으로 돌아갔음
    if (rules_.gap) {
        for (auto& entry : unseenFramesById_) {
            if (geometryById_.find(entry.first) == geometryById_.end()) ++entry.second;
        }
    }
    const auto shouldDropHistory = [&](int trackId) {
        if (geometryById_.find(trackId) != geometryById_.end()) return false;
        if (!rules_.gap) return true;
        const auto unseen = unseenFramesById_.find(trackId);
        return unseen == unseenFramesById_.end() || unseen->second > gapPreserveFrames;
    };
    for (auto iterator = egoLaneStreakById_.begin(); iterator != egoLaneStreakById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = egoLaneStreakById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = egoLaneGraceById_.begin(); iterator != egoLaneGraceById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = egoLaneGraceById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = lateralHistoryById_.begin(); iterator != lateralHistoryById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = lateralHistoryById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = overlapHistoryById_.begin(); iterator != overlapHistoryById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = overlapHistoryById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = unseenFramesById_.begin(); iterator != unseenFramesById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = unseenFramesById_.erase(iterator);
        else ++iterator;
    }
    for (auto iterator = lastGeometryById_.begin(); iterator != lastGeometryById_.end();) {
        if (shouldDropHistory(iterator->first)) iterator = lastGeometryById_.erase(iterator);
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
