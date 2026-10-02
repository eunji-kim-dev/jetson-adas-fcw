#include "adas/RiskAnalyzer.hpp"

#include "perception/Classes.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

// --lead-rule history 의 높이 증가 검사용
// 등속 접근이면 h ∝ 1/Z 이고 Z 는 시간에 선형이므로 1/h 가 시간에 선형임
// 샘플의 1/h 를 시간에 대해 최소제곱 직선으로 맞춰 targetFrame 시점의 1/h 를 돌려줌
// log 높이 회귀 기울기(구간 평균 1/TTC)를 순간값처럼 쓰면 충돌이 가까울수록 예상이 실제보다 작아짐 — 그래서 따로 둠
// 샘플이 2개 미만이거나 시간이 전부 같으면 false
template <typename Samples>
bool predictInverseHeight(const Samples& samples, double fps, int targetFrame, double& inverseHeightOut) {
    if (samples.size() < 2) return false;
    const int firstFrame = samples.front().frameIndex;
    double meanTime = 0.0, meanValue = 0.0;
    for (const auto& sample : samples) {
        meanTime += static_cast<double>(sample.frameIndex - firstFrame) / fps;
        meanValue += 1.0 / std::max(static_cast<double>(sample.boxHeight), 1.0);
    }
    meanTime /= static_cast<double>(samples.size());
    meanValue /= static_cast<double>(samples.size());
    double numerator = 0.0, denominator = 0.0;
    for (const auto& sample : samples) {
        const double timeDifference = static_cast<double>(sample.frameIndex - firstFrame) / fps - meanTime;
        numerator += timeDifference * (1.0 / std::max(static_cast<double>(sample.boxHeight), 1.0) - meanValue);
        denominator += timeDifference * timeDifference;
    }
    if (denominator <= 1.0e-9) return false;
    const double slope = numerator / denominator;
    const double targetTime = static_cast<double>(targetFrame - firstFrame) / fps;
    inverseHeightOut = meanValue + slope * (targetTime - meanTime);
    return true;
}

// --lead-rule bonnet: 보닛 반사 의심 모양 (실험값, 9/30 04·06·07 x86 영상으로 잡음). 셋 다 만족하면 보닛 모양
// 실제 차의 범퍼만 잡힌 박스도 같은 모양이 될 수 있으므로 "의심"이고, 정상 모양이 한 번 나오면 바로 풀림
constexpr float bonnetBottomRatio = 0.95F;    // 아래 변이 화면 높이의 95% 아래
constexpr float bonnetTopRatio = 0.55F;       // 위 변도 화면 높이의 55% 아래 = 박스 전체가 화면 아래쪽
constexpr float bonnetMinimumAspect = 2.5F;   // 폭 ÷ 높이. 04 ≈16, 06 ≈9, 07 ≈4.7

// --lead-rule hold / bonnet: 비정상 관측을 이 프레임 수까지는 리셋하지 않고 보존함 (gap 의 3프레임과 같음)
constexpr int holdLimitFrames = 3;

} // namespace
RiskAnalyzer::RiskAnalyzer(
    double fps,
    int frameHeight,
    int historySize,
    int staleFrameLimit,
    float heightCollapseResetRatio,
    bool bonnetRule,
    bool holdRule
)
    : fps_(fps),

      // 박스 하단이 화면 밖으로 나갔는지 판단하고
      // 경고 최소 박스 높이를 해상도 비율로 계산할 때 씀
      frameHeight_(std::max(frameHeight, 1)),

      // TTC 추세 계산하려면 샘플이 최소한 좀 있어야 해서 8 밑으로는 못 내려가게 막음
      historySize_(static_cast<std::size_t>(std::max(historySize, 8))),

      staleFrameLimit_(std::max(staleFrameLimit, 1)),

      // 같은 ID가 다시 나타나도 공백이 5프레임 넘으면 이전 움직임이랑 안 이어붙임
      // (오래 놓쳤다가 다시 잡히면 박스 크기가 확 달라져서 TTC가 이상하게 튐)
      maximumHistoryGapFrames_(5),

      // 0 이면 끔. 음수는 0 으로 취급함
      heightCollapseResetRatio_(std::max(heightCollapseResetRatio, 0.0F)),
      bonnetRule_(bonnetRule),
      holdRule_(holdRule) {
    if (fps_ <= 0.0) {
        throw std::invalid_argument("RiskAnalyzer FPS must be greater than zero.");
    }
}

// 저장된 샘플들에 선형회귀 돌려서 초당 변화율 뽑아냄
// useLogHeight=true면 log(박스 높이), false면 groundY 기준
// 첫/끝 프레임만 비교 안 하고 전체 샘플로 회귀 돌리는 이유는
// 박스가 한두 프레임 흔들려도 기울기가 크게 안 튀게 하려고
float RiskAnalyzer::calculateRegressionSlope(
    const std::deque<Sample>& samples,
    double fps,
    bool useLogHeight
) {
    // 기울기 내려면 서로 다른 시점 샘플이 최소 2개는 있어야 함
    if (samples.size() < 2) {
        return 0.0F;
    }

    // 절대 프레임 번호 대신 첫 샘플을 0초로 잡고 상대 시간으로 계산 (계산이 단순해짐)
    const int firstFrame = samples.front().frameIndex;

    double meanTime = 0.0;
    double meanValue = 0.0;

    for (const Sample& sample : samples) {
        const double timeSeconds =
            static_cast<double>(sample.frameIndex - firstFrame) / fps;

        // 박스 높이 그대로가 아니라 log(높이)를 쓰는 이유:
        // 전방 물체의 화면상 높이 h는 대략 실제 거리 Z에 반비례함 (h ≈ k / Z)
        // 그럼 log(h) ≈ log(k) - log(Z)라서, log(h)의 변화율이 거리 감소 추세를 더 잘 보여줌
        // boxHeight가 0 이하로 안 내려가게 최소 1로 clamp
        const double value = useLogHeight
            ? std::log(std::max(static_cast<double>(sample.boxHeight), 1.0))
            : static_cast<double>(sample.groundY);

        meanTime += timeSeconds;
        meanValue += value;
    }

    meanTime /= static_cast<double>(samples.size());
    meanValue /= static_cast<double>(samples.size());

    // 최소제곱법 기울기: Σ((t-t̄)(v-v̄)) / Σ((t-t̄)^2)
    double numerator = 0.0;
    double denominator = 0.0;

    for (const Sample& sample : samples) {
        const double timeSeconds =
            static_cast<double>(sample.frameIndex - firstFrame) / fps;

        const double value = useLogHeight
            ? std::log(std::max(static_cast<double>(sample.boxHeight), 1.0))
            : static_cast<double>(sample.groundY);

        const double timeDifference = timeSeconds - meanTime;

        numerator += timeDifference * (value - meanValue);
        denominator += timeDifference * timeDifference;
    }

    // 샘플들 시간이 사실상 다 같으면 분모가 0에 가까워짐 -> 나눗셈 사고 방지
    if (denominator <= 1.0e-9) {
        return 0.0F;
    }

    return static_cast<float>(numerator / denominator);
}

// samples[start, start+count) 구간 박스 높이 평균
// 과거 3개 vs 최근 3개 평균 비교해서 heightGrowthRatio 낼 때 씀 (순간 흔들림 완화용)
float RiskAnalyzer::calculateAverageHeight(
    const std::deque<Sample>& samples,
    std::size_t start,
    std::size_t count
) {
    float total = 0.0F;

    for (std::size_t index = 0; index < count; ++index) {
        total += samples[start + index].boxHeight;
    }

    return total / static_cast<float>(count);
}

// 트랙 기록 리셋. 분석 대상에서 빠졌거나, 공백이 너무 길었거나,
// 과거 기록이랑 지금 기록을 이어 붙이면 안 되는 상황에서 호출됨
// lastSeenFrame은 정리(removeStaleTracks) 판단에 필요해서 여기선 안 건드림
void RiskAnalyzer::resetTrackHistory(TrackHistory& history) {
    history.samples.clear();

    history.stableLevel = RiskLevel::Safe;
    history.pendingLevel = RiskLevel::Safe;

    history.pendingFrames = 0;
    history.lowerLevelFrames = 0;

    // 절단 보정용 종횡비도 같이 버림
    // 이전 트랙 상태에서 재던 값을 새 관측에 이어 쓰면 등가 높이가 어긋남
    history.lastAspectRatio = 0.0F;

    history.heldSamples.clear();
}

// 단계 판정만 SAFE로 되돌리고 샘플 이력은 그대로 둠
// LEAD가 아닌 ego lane 차량은 샘플만 계속 쌓아두는데,
// 예전에 LEAD였을 때 올라간 stableLevel이 남아 있으면
// 지금 판정하지도 않은 단계가 결과로 나가버림
void RiskAnalyzer::clearLevelState(TrackHistory& history) {
    history.stableLevel = RiskLevel::Safe;
    history.pendingLevel = RiskLevel::Safe;

    history.pendingFrames = 0;
    history.lowerLevelFrames = 0;
}

// hold/bonnet 공통: 절단 보정 높이. update() 의 계산과 같은 식 (종횡비는 마지막 비절단 프레임 값)
float RiskAnalyzer::compensatedHeight(const TrackHistory& history, const cv::Rect& box) const {
    const bool truncated = box.y + box.height >= frameHeight_ - 2;
    if (truncated && history.lastAspectRatio > 1.0e-3F) {
        return static_cast<float>(std::max(box.width, 1)) / history.lastAspectRatio;
    }
    return static_cast<float>(std::max(box.height, 1));
}

// history 규칙의 높이 급변 검사. 상태는 바꾸지 않음
// update() 의 리셋 경로와 classifyObservations() 의 보존 경로가 같은 판정을 쓰도록 여기로 뺌
//   감소: 40% 넘게 줄면 급변. 차가 0.33초(5프레임) 안에 1.7배 멀어지는 일은 없음
//   증가: 이전 샘플의 1/h 를 시간에 선형으로 맞춰(등속 접근 모델) 예상 높이를 구하고, max(예상, 직전) 의 1.4배를 넘으면 급변
//         boxHeight 는 절단 보정값이라 화면 높이보다 커지는 게 정상 → 예상에 화면 상한을 두지 않음
//         예측 불가(예상 1/h ≤ 0, 샘플 2개 미만)면 증가로는 급변으로 안 봄 — 충돌 직전 급접근 이력을 지우지 않는 정책
bool RiskAnalyzer::isHeightAnomaly(const TrackHistory& history, float boxHeight, int currentFrame) const {
    if (heightCollapseResetRatio_ <= 0.0F || history.samples.empty()) return false;
    const Sample& previous = history.samples.back();
    if (currentFrame - previous.frameIndex < 1) return false;

    const float previousHeight = std::max(previous.boxHeight, 1.0F);
    if (boxHeight < previousHeight * (1.0F - heightCollapseResetRatio_)) return true;
    if (boxHeight > previousHeight * (1.0F + heightCollapseResetRatio_)) {
        double predictedInverseHeight = 0.0;
        const bool predictionUsable =
            predictInverseHeight(history.samples, fps_, currentFrame, predictedInverseHeight) &&
            predictedInverseHeight > 1.0e-6;
        if (predictionUsable) {
            const float predictedHeight = static_cast<float>(1.0 / predictedInverseHeight);
            // 멀어지던 추세(예상이 직전보다 작음)에서는 직전 관측을 기준으로 둠
            const float expectedHeight = std::max(previousHeight, predictedHeight);
            return boxHeight > expectedHeight * (1.0F + heightCollapseResetRatio_);
        }
    }
    return false;
}

// bonnet: 보닛 반사 의심 모양. 아래 변이 화면 바닥에 닿고, 위 변까지 화면 아래쪽에 있고, 납작함
bool RiskAnalyzer::isBonnetShape(const cv::Rect& box) const {
    const float frameHeight = static_cast<float>(frameHeight_);
    if (static_cast<float>(box.y + box.height) < frameHeight * bonnetBottomRatio) return false;
    if (static_cast<float>(box.y) < frameHeight * bonnetTopRatio) return false;
    const float aspect = static_cast<float>(box.width) / static_cast<float>(std::max(box.height, 1));
    return aspect >= bonnetMinimumAspect;
}

// hold/bonnet: 이번 프레임 관측을 트랙마다 정상 / held / bonnet 확정 / reset 으로 분류함
// 보닛 모양(bonnet)과 높이 급변(hold)을 같은 비정상 관측으로 보고 heldFrames 하나로 셈
//   연속 holdLimitFrames 까지: held — 샘플을 안 넣고 단계·이력을 그대로 둠. LeadSelector 도 이 프레임은 이력을 동결함
//   그 뒤: 보닛 모양이었으면 bonnet 확정, 아니면 reset
//   정상 프레임이 한 번 나오면 카운터가 0 이 되고 확정도 풀림
// 기존 트랙(LEAD 포함)의 박스가 보닛 모양으로 바뀌어도 첫 프레임부터 held 라 옛 streak 로 LEAD 가 되거나 샘플이 들어가는 일이 없음
void RiskAnalyzer::classifyObservations(const std::vector<TrackedObject>& trackedObjects, int currentFrame) {
    observationStates_.clear();
    if (!bonnetRule_ && !holdRule_) return;

    for (const TrackedObject& trackedObject : trackedObjects) {
        if (!isVehicleClass(trackedObject.classId)) continue;
        TrackHistory& history = histories_[trackedObject.trackId];

        ObservationState state;
        state.frame = currentFrame;
        state.referenceBox = trackedObject.box;

        const bool bonnetShape = bonnetRule_ && isBonnetShape(trackedObject.box);
        bool anomalous = bonnetShape;
        // 공백이 5프레임을 넘으면 update() 가 어차피 이력을 지우므로 급변 검사는 이어진 트랙에만 함
        const bool continuous = history.lastSeenFrame >= 0 && currentFrame - history.lastSeenFrame <= maximumHistoryGapFrames_;
        const float boxHeight = compensatedHeight(history, trackedObject.box);
        if (!anomalous && holdRule_ && continuous) {
            anomalous = isHeightAnomaly(history, boxHeight, currentFrame);
        }

        if (anomalous) {
            ++history.heldFrames;
            if (history.heldFrames <= holdLimitFrames) {
                state.held = true;
                if (history.hasLastGoodBox) state.referenceBox = history.lastGoodBox;
                // 보존 프레임의 샘플을 따로 둠. 리셋으로 끝나면 이걸로 새 이력을 시작함 (update() 의 reset 경로)
                const bool truncated = trackedObject.box.y + trackedObject.box.height >= frameHeight_ - 2;
                const float groundY = truncated
                    ? static_cast<float>(trackedObject.box.y) + boxHeight
                    : static_cast<float>(trackedObject.box.y + trackedObject.box.height);
                history.heldSamples.push_back({currentFrame, boxHeight, groundY});
            } else if (bonnetShape) {
                state.bonnet = true;
            } else {
                state.reset = true;
            }
        } else {
            history.heldFrames = 0;
            history.heldSamples.clear();
            history.lastGoodBox = trackedObject.box;
            history.hasLastGoodBox = true;
        }

        history.observation = state;
        observationStates_[trackedObject.trackId] = state;
    }
}

// rawLevel을 그대로 안 쓰고 몇 프레임 지켜본 다음 반영함
// 올라갈 때: SAFE->CAUTION은 4프레임, (SAFE/CAUTION)->DANGER는 2프레임 연속 확인
// 내려갈 때: DANGER였으면 10프레임, CAUTION이었으면 6프레임 동안 낮은 값이 유지돼야 내려감
// 이렇게 안 하면 박스가 한 프레임 흔들릴 때마다 경고가 깜빡깜빡함
RiskLevel RiskAnalyzer::stabilizeLevel(TrackHistory& history, RiskLevel rawLevel) const {
    // 이미 같은 단계면 딱히 승격/강등 처리할 게 없음
    if (rawLevel == history.stableLevel) {
        history.pendingLevel = rawLevel;

        history.pendingFrames = 0;
        history.lowerLevelFrames = 0;

        return history.stableLevel;
    }

    // Safe=0, Caution=1, Danger=2 라서 정수 비교로 고저 판단
    const int rawValue = static_cast<int>(rawLevel);
    const int stableValue = static_cast<int>(history.stableLevel);

    // 현재보다 높은 값이면 승격 후보
    if (rawValue > stableValue) {
        history.lowerLevelFrames = 0;

        if (history.pendingLevel == rawLevel) {
            // 이전이랑 같은 후보가 계속 나오면 카운트 증가
            ++history.pendingFrames;
        } else {
            // 다른 후보가 나오면 새로 잡고 카운트 1부터 시작
            history.pendingLevel = rawLevel;
            history.pendingFrames = 1;
        }

        // DANGER는 빨리 반응해야 하니 2프레임, CAUTION은 오경고 줄이려고 4프레임
        const int requiredFrames = rawLevel == RiskLevel::Danger ? 2 : 4;

        if (history.pendingFrames >= requiredFrames) {
            history.stableLevel = rawLevel;
            history.pendingFrames = 0;
        }

        return history.stableLevel;
    }

    // 여기 왔으면 rawLevel이 현재보다 낮은 경우
    // 한 프레임 좋아졌다고 바로 경고 꺼버리지 않고 낮은 값이 지속되는지 확인
    history.pendingFrames = 0;
    ++history.lowerLevelFrames;

    const int requiredFrames = history.stableLevel == RiskLevel::Danger ? 10 : 6;

    if (history.lowerLevelFrames >= requiredFrames) {
        history.stableLevel = rawLevel;
        history.pendingLevel = rawLevel;
        history.lowerLevelFrames = 0;
    }

    return history.stableLevel;
}

RiskResult RiskAnalyzer::update(
    const TrackedObject& trackedObject,
    bool isAnalysisTarget,
    bool isLeadTarget,
    int currentFrame
) {
    // trackId로 기록 꺼내옴, 처음 보는 ID면 기본값으로 새로 생김
    TrackHistory& history = histories_[trackedObject.trackId];

    // --diag-log: 이번 프레임에 샘플을 지운 이유. 지울 샘플이 있었을 때만 적음 (판정에는 안 씀)
    const char* resetReason = "";
    auto noteReset = [&](const char* reason) {
        if (!history.samples.empty()) resetReason = reason;
    };

    // 같은 ID라도 검출 공백이 너무 길었으면 예전 움직임이랑 지금 움직임을 안 이어붙임
    if (history.lastSeenFrame >= 0 &&
        currentFrame - history.lastSeenFrame > maximumHistoryGapFrames_) {
        noteReset("gap");
        resetTrackHistory(history);
    }

    history.lastSeenFrame = currentFrame;

    // 분석 대상(ego lane 안의 차량)이 아니면 기록 자체를 버림
    // 차선 밖으로 나간 차량의 옛날 박스 변화를 나중에 이어 쓰면 TTC가 엉뚱하게 나옴
    if (!isAnalysisTarget) {
        noteReset("not_target");
        resetTrackHistory(history);

        RiskResult result;
        result.historyReset = resetReason;
        return result;
    }

    // hold/bonnet: classifyObservations() 의 이번 프레임 판정을 따름
    if ((bonnetRule_ || holdRule_) && history.observation.frame == currentFrame) {
        const ObservationState& observation = history.observation;
        if (observation.bonnet) {
            // 보닛 확정: 분석 대상에서 빼고 이력을 지움 (LeadSelector 도 후보에서 뺌)
            noteReset("bonnet");
            resetTrackHistory(history);
            RiskResult result;
            result.historyReset = resetReason;
            return result;
        }
        if (observation.held) {
            // 보존: 샘플을 안 넣고 직전 단계를 그대로 돌려줌. 안정화(stabilizeLevel)도 리셋도 안 거침
            // DANGER 였으면 DANGER 그대로. 정상 복귀 프레임에서 원시 단계가 낮으면 기존 10프레임 강등이 작동함
            if (!isLeadTarget) clearLevelState(history);
            RiskResult result;
            result.level = history.stableLevel;
            result.sampleCount = static_cast<int>(history.samples.size());
            result.observationHeld = true;
            // --diag-log: 샘플에 안 넣은 이번 관측의 보정 높이와 종횡비를 남김
            result.boxHeight = compensatedHeight(history, trackedObject.box);
            result.aspectRatio = history.lastAspectRatio;
            result.historyReset = resetReason;
            return result;
        }
        if (observation.reset) {
            // 보존 한도를 넘긴 급변 = 변화가 계속되는 것. 보존했던 프레임의 샘플로 새 이력을 시작함
            // (즉시 리셋하던 history 와 같은 시점에 샘플 10개가 차게. 08: 135 급변 → 이전은 144, 보존만 하면 147)
            const std::vector<Sample> carried = history.heldSamples;
            noteReset("hold_limit");
            resetTrackHistory(history);
            for (const Sample& sample : carried) history.samples.push_back(sample);
        }
    }

    const int rawBottom = trackedObject.box.y + trackedObject.box.height;
    const int rawWidth = std::max(trackedObject.box.width, 1);
    const int rawHeight = std::max(trackedObject.box.height, 1);

    /*
     * 검출 박스는 detectObjects()에서 프레임 경계로 clamp되니까,
     * 선행 차량이 아주 가까워져서 박스 하단이 화면 밖으로 나가면
     * height 증가랑 groundY 하강이 동시에 멈춰버림
     *
     * 하필 위험이 제일 큰 순간에 두 지표가 다 0으로 수렴해서
     * TTC-P가 무한대가 되고 경고가 SAFE로 되돌아감
     *
     * 화면 폭 안에 남아 있는 width는 계속 커지니까,
     * 절단 직전 종횡비로 등가 높이를 만들어서 시계열을 이어붙임
     *
     * log(w)랑 log(h)는 기울기가 같고 상수 오프셋만 다른데,
     * 오프셋 제거 없이 그냥 갈아타면 시계열 중간에 계단이 생겨서
     * 없던 기울기가 만들어짐 -> 반드시 종횡비로 환산해야 함
     */
    const bool truncated = rawBottom >= frameHeight_ - 2;

    float boxHeight = static_cast<float>(rawHeight);

    if (!truncated) {
        history.lastAspectRatio =
            static_cast<float>(rawWidth) / static_cast<float>(rawHeight);
    } else if (history.lastAspectRatio > 1.0e-3F) {
        boxHeight = static_cast<float>(rawWidth) / history.lastAspectRatio;
    }

    // 박스 하단 y좌표. 차량이 화면 아래로 내려올수록 커짐
    // 절단된 상태면 실제 하단이 화면 밖이라 719 같은 값에 고정되니까,
    // 등가 높이로 가상 접지점을 만들어서 회귀 입력이 끊기지 않게 함
    const float groundY = truncated
        ? static_cast<float>(trackedObject.box.y) + boxHeight
        : static_cast<float>(rawBottom);

    // --lead-rule history: 마지막 관측 대비 박스 높이가 설명이 안 되게 튀면 이력 전체(샘플 + 안정화 상태)를 초기화함
    // 08번 147 프레임 362→131 같은 붕괴를 이전 추세에 이어 붙이면 TTC 가 엉뚱하게 나옴
    // 감소와 증가를 다르게 봄
    //   감소: 40% 넘게 줄면 붕괴로 봄. 차가 0.33초(5프레임) 안에 1.7배 멀어지는 일은 없음
    //   증가: 급접근이면 정확한 박스도 빠르게 커짐 (TTC 0.4초에서 3프레임 뒤 2배가 등속 접근의 정상값)
    //         그래서 이전 샘플의 1/h 를 시간에 선형으로 맞춰(등속 접근 모델) 지금 시점의 예상 높이를 구하고,
    //         관측 높이가 max(예상, 직전 관측)보다 40% 넘게 크면 초기화함
    //         boxHeight 는 절단 보정값(폭 ÷ 절단 직전 종횡비)이라 화면 높이보다 커지는 게 정상 → 예상에 화면 상한을 두지 않음
    //         예측 불가(예상 1/h ≤ 0 = 모델상 충돌 시점을 지남, 또는 샘플 2개 미만)면 증가 검사는 생략하고 감소 검사만 둠
    //         — 충돌 직전의 급접근 이력을 지우지 않는 쪽을 우선한 정책임. 이 구간에서 다른 차로 ID 가 옮겨 붙는 경우는 못 거름
    // 짧은 공백(1~5프레임) 뒤 재등장한 프레임도 같은 기준으로 검사함 — 공백을 이어 쓰는 gap 규칙에서 재등장 프레임이 잘못된 박스가 섞이는 자리임
    // 5프레임 넘는 공백은 위 maximumHistoryGapFrames_ 에서 이미 초기화됨. 40% 는 큰 붕괴만 걸러내는 실험값임
    // 절단 보정용 종횡비는 이번 프레임 값으로 다시 채움
    // 판정 자체는 isHeightAnomaly() 로 뺐음. hold 규칙이 켜져 있으면 classifyObservations() 가 이미 판정해서(보존/리셋) 여기서는 건너뜀
    if (!holdRule_ && isHeightAnomaly(history, boxHeight, currentFrame)) {
        noteReset("anomaly");
        resetTrackHistory(history);
        if (!truncated) {
            history.lastAspectRatio =
                static_cast<float>(rawWidth) / static_cast<float>(rawHeight);
        }
    }

    // update()가 같은 프레임에 두 번 불리는 경우 대비 - 중복 추가 말고 덮어씀
    if (!history.samples.empty() &&
        history.samples.back().frameIndex == currentFrame) {
        history.samples.back() = {currentFrame, boxHeight, groundY};
    } else {
        history.samples.push_back({currentFrame, boxHeight, groundY});
    }

    // 최대 보관 개수 넘으면 오래된 것부터 버림
    while (history.samples.size() > historySize_) {
        history.samples.pop_front();
    }

    // LEAD가 아니면 샘플만 쌓아두고 단계 판정은 안 함
    // 이렇게 해두면 나중에 LEAD로 뽑히는 순간 이미 샘플이 차 있어서
    // 8~10샘플 기다리는 시간만큼 경고가 늦어지지 않음
    if (!isLeadTarget) {
        clearLevelState(history);
    }

    RiskResult result;

    result.sampleCount = static_cast<int>(history.samples.size());
    result.truncated = truncated;
    // --diag-log 용 관측값 (판정에는 안 씀)
    result.boxHeight = boxHeight;
    result.groundY = groundY;
    result.aspectRatio = history.lastAspectRatio;
    result.historyReset = resetReason;

    // 새 단계가 아직 확정 안 됐어도 일단 지금 안정화된 단계를 기본값으로
    // (LEAD가 아니면 바로 위에서 SAFE로 되돌려놨음)
    result.level = history.stableLevel;

    // 샘플이 8개 미만이면 박스 흔들림인지 진짜 확대 추세인지 구분 안 되니까
    // 이 구간에서는 TTC 계산 없이 SAFE 유지
    if (history.samples.size() < 8) {
        return result;
    }

    // 과거/최근 각각 최대 3개 샘플로 평균 냄
    const std::size_t averageCount =
        std::min<std::size_t>(3, history.samples.size() / 2);

    const float firstHeight =
        calculateAverageHeight(history.samples, 0, averageCount);

    const float lastHeight = calculateAverageHeight(
        history.samples,
        history.samples.size() - averageCount,
        averageCount
    );

    // 과거 평균 대비 최근 평균 비율. 1.06이면 최근에 6% 커졌다는 뜻
    const float heightGrowthRatio = lastHeight / std::max(firstHeight, 1.0F);

    // log(박스 높이)의 초당 증가율. 클수록 빠르게 커지는 중
    const float logHeightRate = calculateRegressionSlope(history.samples, fps_, true);

    // 박스 하단 y좌표의 초당 이동속도. 양수면 화면 아래로 이동중
    const float groundSpeed = calculateRegressionSlope(history.samples, fps_, false);

    // 접근 안 하거나 계산 불가하면 무한대 유지
    float ttcSeconds = std::numeric_limits<float>::infinity();

    // 박스 높이가 거의 안 변하는 구간에서 노이즈성 TTC 튀는 거 막으려고
    // logHeightRate가 0.02보다 클 때만 계산함
    // 전방 물체 화면상 높이가 실제 거리랑 반비례한다고 가정하면
    // d(log(height))/dt ≈ 1/TTC 이므로 TTC-P ≈ 1/logHeightRate
    // (카메라 보정된 실거리 기반 정식 TTC 아니라 어디까지나 근사치)
    if (logHeightRate > 0.02F) {
        ttcSeconds = 1.0F / logHeightRate;
    }

    // 샘플 충분하고 TTC가 유한할 때만 valid=true
    // (샘플은 충분한데 차가 안 가까워지면 TTC는 무한대라 화면엔 "--"로 표시됨)
    result.valid = std::isfinite(ttcSeconds);

    result.ttcSeconds = ttcSeconds;
    result.logHeightRatePerSecond = logHeightRate;
    result.groundSpeedPixelsPerSecond = groundSpeed;
    result.heightGrowthRatio = heightGrowthRatio;

    // LEAD 가 아닌 트랙의 조기 반환은 아래 rawLevel 계산 뒤로 옮김 (--diag-log 용. 판정 결과는 같음)

    // 너무 작은(멀리 있는) 객체의 불안정한 변화율을 위험으로 오판하지 않기 위한
    // 최소 크기 조건에 쓸 현재 프레임 박스 높이
    const float currentHeight = history.samples.back().boxHeight;

    /*
     * 경고 판정 공통 조건
     *
     * 원래 최소 박스 높이가 12/18px이었는데, 원거리 소형 박스가
     * 2~3px 흔들리는 것만으로 TTC-P가 4초대로 계산되는 오경고가 있었음
     * (동영상 14.98~15.58초 구간)
     *
     * 그때는 main.cpp 쪽에 36px 게이트를 하나 더 얹어서 막았는데,
     * 그러면 안쪽 12/18px이 절대 안 걸리는 죽은 값이 됨
     * 그래서 기준을 여기 한 곳으로 합침
     */
    const int minimumWarningHeight =
        std::max(36, static_cast<int>(std::round(frameHeight_ * 0.05)));

    constexpr std::size_t minimumSamplesForWarning = 10;

    const bool baseConditionsMet =
        std::isfinite(ttcSeconds) &&
        history.samples.size() >= minimumSamplesForWarning &&
        currentHeight >= static_cast<float>(minimumWarningHeight);

    // 접지점이 아래로 내려가는 건 접근의 보조 증거인데,
    // 박스가 절단된 상태면 접지점이 화면 하단에 고정돼서 무조건 0이 나옴
    // 절단은 이미 아주 가까워졌다는 뜻이니까 이 조건은 통과시킴
    const bool groundEvidenceForDanger = truncated || groundSpeed >= 8.0F;
    const bool groundEvidenceForCaution = truncated || groundSpeed >= 2.5F;

    // 아직 안정화 전, 이번 프레임만 놓고 본 위험 단계
    RiskLevel rawLevel = RiskLevel::Safe;

    // DANGER: 높이 6% 이상 증가 + 접지점 초당 8px 이상 하강 + TTC 2.5초 이하
    // 여러 조건을 같이 요구하는 이유는 박스 하나만 순간적으로 커지는 오검출 걸러내려고
    if (baseConditionsMet &&
        heightGrowthRatio >= 1.06F &&
        groundEvidenceForDanger &&
        ttcSeconds <= 2.5F) {
        rawLevel = RiskLevel::Danger;
    } else if (baseConditionsMet &&
               heightGrowthRatio >= 1.03F &&
               groundEvidenceForCaution &&
               ttcSeconds <= 5.0F) {
        // CAUTION은 DANGER보다 완화된 기준 (3% / 2.5px/s / 5초)
        rawLevel = RiskLevel::Caution;
    }

    // --diag-log 용 원시 단계. LEAD 가 아닌 트랙은 "LEAD 였다면" 값이고 판정에는 안 씀
    result.rawLevel = rawLevel;

    // LEAD가 아니면 수치만 채워서 돌려주고 단계 판정은 여기서 끝
    // stabilizeLevel을 돌려버리면 LEAD 아닌 동안 내부 단계가 몰래 올라가 있다가
    // LEAD로 바뀌는 순간 승격 카운트 없이 경고가 튀어나옴
    if (!isLeadTarget) {
        return result;
    }

    // 이번 프레임 rawLevel 그대로 안 쓰고 안정화된 값으로 반환
    result.level = stabilizeLevel(history, rawLevel);

    return result;
}

// 오래 안 보인 트랙 기록 삭제
// update()가 안 불리는 객체는 lastSeenFrame이 안 갱신되니까
// currentFrame과의 차이가 staleFrameLimit_ 넘으면 삭제 대상
void RiskAnalyzer::removeStaleTracks(int currentFrame) {
    // 순회하면서 바로 erase하면 iterator 깨지니까 삭제할 것만 먼저 모아둠
    std::vector<int> staleIds;

    for (const auto& entry : histories_) {
        const int trackId = entry.first;
        const TrackHistory& history = entry.second;

        if (history.lastSeenFrame >= 0 &&
            currentFrame - history.lastSeenFrame > staleFrameLimit_) {
            staleIds.push_back(trackId);
        }
    }

    for (const int trackId : staleIds) {
        histories_.erase(trackId);
    }
}

// 장면 전환처럼 이전 프레임과의 연속성이 완전히 끊긴 경우에 호출
// 0.5초 분석 유예만으로는 이번 프레임에 보이는 객체 기록만 지워지고,
// 컷 직전에 사라진 트랙 기록은 staleFrameLimit_까지 남아 있음
void RiskAnalyzer::reset() {
    histories_.clear();
    observationStates_.clear();
}

std::string RiskAnalyzer::toString(RiskLevel level) {
    switch (level) {
        case RiskLevel::Safe: return "SAFE";
        case RiskLevel::Caution: return "CAUTION";
        case RiskLevel::Danger: return "DANGER";
    }

    // 여기 오면 안 되지만 혹시 모르니 기본값
    return "UNKNOWN";
}