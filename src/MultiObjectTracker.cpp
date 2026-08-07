#include "MultiObjectTracker.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

MultiObjectTracker::TrackState::TrackState(const Detection& detection)
    : id(0),
      classId(detection.classId),
      confidence(detection.confidence),
      box(detection.box),
      kalman(8, 4, 0, CV_32F),
      age(1),
      hits(1),
      consecutiveHits(1),
      missedFrames(0),
      confirmed(false) {
    /*
     * Kalman Filter 상태값 8개
     *
     * 0: 중심 x        1: 중심 y
     * 2: x 방향 속도   3: y 방향 속도
     * 4: 박스 너비     5: 박스 높이
     * 6: 너비 변화속도 7: 높이 변화속도
     */
    kalman.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F
    );

    // YOLO에서 직접 얻는 측정값: 중심 x, 중심 y, 박스 너비, 박스 높이
    kalman.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);

    kalman.measurementMatrix.at<float>(0, 0) = 1.0F;
    kalman.measurementMatrix.at<float>(1, 1) = 1.0F;
    kalman.measurementMatrix.at<float>(2, 4) = 1.0F;
    kalman.measurementMatrix.at<float>(3, 5) = 1.0F;

    /*
     * 위치보다 속도와 크기 변화속도의 불확실성을 크게 설정
     *
     * 블랙박스 영상은 카메라 자체가 이동하기 때문에
     * 객체 중심과 박스 크기가 빠르게 달라질 수 있음
     */
    cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(1.0e-2));

    kalman.processNoiseCov.at<float>(2, 2) = 1.0F;
    kalman.processNoiseCov.at<float>(3, 3) = 1.0F;
    kalman.processNoiseCov.at<float>(6, 6) = 0.5F;
    kalman.processNoiseCov.at<float>(7, 7) = 0.5F;

    cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(10.0));

    const float centerX = detection.box.x + detection.box.width / 2.0F;
    const float centerY = detection.box.y + detection.box.height / 2.0F;

    kalman.statePost = (cv::Mat_<float>(8, 1) <<
        centerX,
        centerY,
        0.0F,
        0.0F,
        static_cast<float>(detection.box.width),
        static_cast<float>(detection.box.height),
        0.0F,
        0.0F
    );

    kalman.statePre = kalman.statePost.clone();
}

MultiObjectTracker::MultiObjectTracker(
    float highConfidenceThreshold,
    float lowConfidenceThreshold,
    int minConfirmationHits,
    int maxMissedFrames
)
    : highConfidenceThreshold_(highConfidenceThreshold),
      lowConfidenceThreshold_(lowConfidenceThreshold),
      minConfirmationHits_(minConfirmationHits),
      maxMissedFrames_(maxMissedFrames),
      nextTrackId_(1) {
}

float MultiObjectTracker::calculateIoU(
    const cv::Rect2f& first,
    const cv::Rect2f& second
) {
    const cv::Rect2f intersection = first & second;
    const float intersectionArea = intersection.area();

    if (intersectionArea <= 0.0F) {
        return 0.0F;
    }

    const float unionArea = first.area() + second.area() - intersectionArea;

    if (unionArea <= 0.0F) {
        return 0.0F;
    }

    return intersectionArea / unionArea;
}

float MultiObjectTracker::calculateCenterDistance(
    const cv::Rect2f& first,
    const cv::Rect2f& second
) {
    const float firstCenterX = first.x + first.width / 2.0F;
    const float firstCenterY = first.y + first.height / 2.0F;

    const float secondCenterX = second.x + second.width / 2.0F;
    const float secondCenterY = second.y + second.height / 2.0F;

    const float differenceX = firstCenterX - secondCenterX;
    const float differenceY = firstCenterY - secondCenterY;

    return std::sqrt(differenceX * differenceX + differenceY * differenceY);
}

float MultiObjectTracker::calculateAreaRatio(
    const cv::Rect2f& first,
    const cv::Rect2f& second
) {
    const float firstArea = std::max(first.area(), 1.0F);
    const float secondArea = std::max(second.area(), 1.0F);

    return std::max(firstArea, secondArea) / std::min(firstArea, secondArea);
}

float MultiObjectTracker::calculateAllowedCenterDistance(
    const cv::Rect2f& first,
    const cv::Rect2f& second
) {
    const float firstDiagonal =
        std::sqrt(first.width * first.width + first.height * first.height);

    const float secondDiagonal =
        std::sqrt(second.width * second.width + second.height * second.height);

    const float largestDiagonal = std::max(firstDiagonal, secondDiagonal);

    /*
     * 기존 상한 100px은 가까운 대형 차량에 너무 작았음
     *
     * 박스 대각선에 비례해 허용 거리를 늘리고
     * 다른 차량까지 연결되는 것을 막기 위해 최대값은 320px로 설정
     */
    return std::clamp(largestDiagonal * 0.85F + 8.0F, 16.0F, 320.0F);
}

cv::Rect2f MultiObjectTracker::stateToRect(const cv::Mat& state) {
    const float centerX = state.at<float>(0);
    const float centerY = state.at<float>(1);

    const float width = std::max(state.at<float>(4), 1.0F);
    const float height = std::max(state.at<float>(5), 1.0F);

    return cv::Rect2f(
        centerX - width / 2.0F,
        centerY - height / 2.0F,
        width,
        height
    );
}

cv::Mat MultiObjectTracker::detectionToMeasurement(const Detection& detection) {
    const float centerX = detection.box.x + detection.box.width / 2.0F;
    const float centerY = detection.box.y + detection.box.height / 2.0F;

    return (cv::Mat_<float>(4, 1) <<
        centerX,
        centerY,
        static_cast<float>(detection.box.width),
        static_cast<float>(detection.box.height)
    );
}

bool MultiObjectTracker::isSameTrackingCategory(int firstClassId, int secondClassId) {
    const auto isFourWheelVehicle = [](int classId) {
        return classId == 2 || classId == 5 || classId == 7;
    };

    const auto isTwoWheelVehicle = [](int classId) {
        return classId == 1 || classId == 3;
    };

    if (isFourWheelVehicle(firstClassId) && isFourWheelVehicle(secondClassId)) {
        return true;
    }

    if (isTwoWheelVehicle(firstClassId) && isTwoWheelVehicle(secondClassId)) {
        return true;
    }

    return firstClassId == secondClassId;
}

std::vector<TrackedObject> MultiObjectTracker::update(
    const std::vector<Detection>& detections
) {
    // YOLO 결과를 고신뢰도와 저신뢰도로 분리
    std::vector<int> highDetectionIndices;
    std::vector<int> lowDetectionIndices;

    for (int detectionIndex = 0;
         detectionIndex < static_cast<int>(detections.size());
         ++detectionIndex) {
        const float confidence = detections[detectionIndex].confidence;

        if (confidence >= highConfidenceThreshold_) {
            highDetectionIndices.push_back(detectionIndex);
        } else if (confidence >= lowConfidenceThreshold_) {
            lowDetectionIndices.push_back(detectionIndex);
        }
    }

    // 기존 Track의 다음 위치와 크기를 예측
    for (TrackState& track : tracks_) {
        const cv::Mat predictedState = track.kalman.predict();

        track.box = stateToRect(predictedState);

        ++track.age;
        ++track.missedFrames;
    }

    const int previousTrackCount = static_cast<int>(tracks_.size());

    std::vector<bool> matchedTracks(tracks_.size(), false);
    std::vector<bool> matchedDetections(detections.size(), false);

    enum class MatchPass {
        RecentHigh,
        RecentLow,
        RecoveryHigh
    };

    struct MatchCandidate {
        float cost;
        int trackIndex;
        int detectionIndex;
    };

    const auto performMatching = [&](
        const std::vector<int>& trackIndices,
        const std::vector<int>& detectionIndices,
        MatchPass matchPass
    ) {
        std::vector<MatchCandidate> candidates;

        for (const int trackIndex : trackIndices) {
            if (matchedTracks[trackIndex]) {
                continue;
            }

            const TrackState& track = tracks_[trackIndex];

            // 초기 Track은 아직 속도를 충분히 학습하지 못한 상태
            // confirmed가 아니거나 hits가 확인 프레임 수보다 작으면 초기 Track
            const bool immatureTrack =
                !track.confirmed || track.hits < minConfirmationHits_;

            for (const int detectionIndex : detectionIndices) {
                if (matchedDetections[detectionIndex]) {
                    continue;
                }

                const Detection& detection = detections[detectionIndex];

                if (!isSameTrackingCategory(track.classId, detection.classId)) {
                    continue;
                }

                const cv::Rect2f detectionBox(detection.box);

                const float iou = calculateIoU(track.box, detectionBox);

                const float centerDistance =
                    calculateCenterDistance(track.box, detectionBox);

                const float allowedDistance =
                    calculateAllowedCenterDistance(track.box, detectionBox);

                const float areaRatio = calculateAreaRatio(track.box, detectionBox);

                float iouGate = 0.05F;
                float distanceMultiplier = 1.20F;
                float maximumAreaRatio = 3.5F;

                if (matchPass == MatchPass::RecentHigh) {
                    // 새 Track은 속도 0으로 시작하므로
                    // 처음 2~3회의 매칭에서는 거리 게이트를 넓게 적용
                    if (immatureTrack) {
                        iouGate = 0.01F;
                        distanceMultiplier = 2.50F;
                        maximumAreaRatio = 5.0F;
                    }
                } else if (matchPass == MatchPass::RecentLow) {
                    // 저신뢰도 검출은 기존 Track 유지용
                    // 초기 Track도 재시도 기회를 받지만 고신뢰도보다 조금 엄격하게 판단
                    iouGate = 0.03F;
                    distanceMultiplier = immatureTrack ? 2.00F : 1.10F;
                    maximumAreaRatio = immatureTrack ? 4.5F : 3.0F;
                } else {
                    // 여러 프레임 놓친 확정 Track
                    // 놓친 프레임 수가 늘수록 이동 가능한 거리도 늘어나므로
                    // 허용 거리를 단계적으로 확장
                    iouGate = 0.03F;

                    distanceMultiplier = std::min(
                        1.0F + static_cast<float>(track.missedFrames - 1) * 0.35F,
                        2.60F
                    );

                    maximumAreaRatio = 4.0F;
                }

                /*
                 * 가장 중요한 수정:
                 * IoU가 기준 이상이면 중심점 거리가 크더라도
                 * 거리 조건이 해당 후보를 다시 거부하지 않음
                 * 거리 게이트는 IoU가 낮을 때만 보조 수단으로 사용
                 */
                const bool passedByIoU = iou >= iouGate;

                const float distanceLimit = allowedDistance * distanceMultiplier;

                if (!passedByIoU && centerDistance > distanceLimit) {
                    continue;
                }

                // 박스 면적 차이도 IoU가 낮은 경우에만 하드 게이트로 사용
                // IoU가 충분히 높다면 거리나 면적 조건이 마지막에 매칭을 뒤집지 않음
                if (!passedByIoU && areaRatio > maximumAreaRatio) {
                    continue;
                }

                const float normalizedDistance = std::min(
                    centerDistance / std::max(distanceLimit, 1.0F),
                    2.0F
                );

                const float sizeDifference = std::min(
                    std::abs(std::log(areaRatio)) / std::log(maximumAreaRatio),
                    1.0F
                );

                float cost =
                    (1.0F - iou) * 0.58F +
                    normalizedDistance * 0.27F +
                    sizeDifference * 0.15F;

                // 여러 프레임 놓친 과거 ID에는 감점
                // 정상적으로 이어지고 있는 ID가 오래된 ID보다 우선 연결되도록 함
                if (matchPass == MatchPass::RecoveryHigh) {
                    cost += static_cast<float>(track.missedFrames - 1) * 0.08F;
                }

                // 같은 차량이 car와 truck 사이에서 분류가 바뀌는 것은 허용하되,
                // 완전히 같은 classId인 후보를 조금 우선
                if (track.classId != detection.classId) {
                    cost += 0.03F;
                }

                candidates.push_back({cost, trackIndex, detectionIndex});
            }
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const MatchCandidate& first, const MatchCandidate& second) {
                return first.cost < second.cost;
            }
        );

        for (const MatchCandidate& candidate : candidates) {
            if (matchedTracks[candidate.trackIndex] ||
                matchedDetections[candidate.detectionIndex]) {
                continue;
            }

            TrackState& track = tracks_[candidate.trackIndex];
            const Detection& detection = detections[candidate.detectionIndex];

            track.kalman.correct(detectionToMeasurement(detection));

            // 현재 프레임의 위치와 크기는 실제 YOLO 박스에 맞춤
            // 속도와 크기 변화속도는 Kalman Filter가 계속 추정한 값을 유지
            const float centerX = detection.box.x + detection.box.width / 2.0F;
            const float centerY = detection.box.y + detection.box.height / 2.0F;

            track.kalman.statePost.at<float>(0) = centerX;
            track.kalman.statePost.at<float>(1) = centerY;

            track.kalman.statePost.at<float>(4) =
                static_cast<float>(detection.box.width);

            track.kalman.statePost.at<float>(5) =
                static_cast<float>(detection.box.height);

            track.box = cv::Rect2f(detection.box);
            track.classId = detection.classId;
            track.confidence = detection.confidence;

            track.missedFrames = 0;
            ++track.hits;

            const bool lowConfidenceMatch = matchPass == MatchPass::RecentLow;

            // 저신뢰도 결과는 Track을 유지하는 데 사용하지만,
            // 미확정 Track을 정식 ID로 승격시키는 근거로는 사용하지 않음
            if (!lowConfidenceMatch) {
                ++track.consecutiveHits;
            }

            if (!track.confirmed &&
                !lowConfidenceMatch &&
                track.consecutiveHits >= minConfirmationHits_) {
                track.confirmed = true;
                track.id = nextTrackId_;
                ++nextTrackId_;
            }

            matchedTracks[candidate.trackIndex] = true;
            matchedDetections[candidate.detectionIndex] = true;
        }
    };

    // predict() 후 missedFrames == 1이면 직전 프레임까지 검출됐던 Track
    // 확정/미확정 Track을 모두 포함
    std::vector<int> recentTrackIndices;

    for (int trackIndex = 0; trackIndex < previousTrackCount; ++trackIndex) {
        if (tracks_[trackIndex].missedFrames == 1) {
            recentTrackIndices.push_back(trackIndex);
        }
    }

    // 1차: 최근 Track과 고신뢰도 검출을 연결
    // 초기 Track은 이 단계에서 완화된 거리 조건을 사용
    performMatching(recentTrackIndices, highDetectionIndices, MatchPass::RecentHigh);

    /*
     * 2차:
     * 1차에서 실패한 missedFrames == 1 Track을 저신뢰도 검출과 다시 연결
     * 기존 코드와 달리 확정 Track뿐 아니라 미확정 초기 Track도 재시도 기회를 받음
     */
    std::vector<int> unmatchedRecentTracks;

    for (const int trackIndex : recentTrackIndices) {
        if (!matchedTracks[trackIndex]) {
            unmatchedRecentTracks.push_back(trackIndex);
        }
    }

    performMatching(unmatchedRecentTracks, lowDetectionIndices, MatchPass::RecentLow);

    // 3차: 2프레임 이상 놓친 확정 Track을 남은 고신뢰도 검출과 재연결
    std::vector<int> lostConfirmedTracks;

    const int maximumRecoveryFrames = std::min(maxMissedFrames_, 10);

    for (int trackIndex = 0; trackIndex < previousTrackCount; ++trackIndex) {
        const TrackState& track = tracks_[trackIndex];

        if (track.confirmed &&
            !matchedTracks[trackIndex] &&
            track.missedFrames >= 2 &&
            track.missedFrames <= maximumRecoveryFrames) {
            lostConfirmedTracks.push_back(trackIndex);
        }
    }

    performMatching(lostConfirmedTracks, highDetectionIndices, MatchPass::RecoveryHigh);

    // 이번 프레임에서 연결되지 않은 기존 Track은 연속 검출 횟수를 초기화
    for (int trackIndex = 0; trackIndex < previousTrackCount; ++trackIndex) {
        if (!matchedTracks[trackIndex]) {
            tracks_[trackIndex].consecutiveHits = 0;
        }
    }

    // 동영상 17초 ID:29 0.26 오류 수정
    // 새 트랙 생성 기준은 유지 기준보다 살짝 높게 잡음
    //   0.25~0.30 : 기존 트랙 매칭에만 사용 (ID 유지용)
    //   0.30 이상 : 신규 트랙 생성 허용
    // 0.26짜리 순간 오검출이 새 ID로 인식되는 것을 방지
    constexpr float newTrackConfidenceThreshold = 0.30F;

    for (const int detectionIndex : highDetectionIndices) {
        if (matchedDetections[detectionIndex]) {
            continue;
        }

        const Detection& detection = detections[detectionIndex];

        if (detection.confidence < newTrackConfidenceThreshold) {
            continue;
        }

        tracks_.emplace_back(detection);

        TrackState& newTrack = tracks_.back();

        if (minConfirmationHits_ <= 1) {
            newTrack.confirmed = true;
            newTrack.id = nextTrackId_;
            ++nextTrackId_;
        }
    }

    // 미확정 Track은 두 프레임 연속 놓치면 삭제
    // 확정 Track은 maxMissedFrames까지 상태를 보관
    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [this](const TrackState& track) {
                if (!track.confirmed) {
                    return track.missedFrames > 1;
                }

                return track.missedFrames > maxMissedFrames_;
            }
        ),
        tracks_.end()
    );

    std::vector<TrackedObject> trackedObjects;

    // 정식으로 확정됐고, 현재 프레임에서 실제 검출과 연결된 객체만 표시
    for (const TrackState& track : tracks_) {
        if (!track.confirmed || track.missedFrames != 0) {
            continue;
        }

        trackedObjects.push_back({
            track.id,
            track.classId,
            track.confidence,
            cv::Rect(
                static_cast<int>(std::round(track.box.x)),
                static_cast<int>(std::round(track.box.y)),
                std::max(static_cast<int>(std::round(track.box.width)), 1),
                std::max(static_cast<int>(std::round(track.box.height)), 1)
            )
        });
    }

    return trackedObjects;
}