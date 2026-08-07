#include "RiskAnalyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

RiskAnalyzer::RiskAnalyzer(
    double fps,
    int frameHeight,
    int historySize,
    int staleFrameLimit
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
      maximumHistoryGapFrames_(5) {
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

    // 같은 ID라도 검출 공백이 너무 길었으면 예전 움직임이랑 지금 움직임을 안 이어붙임
    if (history.lastSeenFrame >= 0 &&
        currentFrame - history.lastSeenFrame > maximumHistoryGapFrames_) {
        resetTrackHistory(history);
    }

    history.lastSeenFrame = currentFrame;

    // 분석 대상(ego lane 안의 차량)이 아니면 기록 자체를 버림
    // 차선 밖으로 나간 차량의 옛날 박스 변화를 나중에 이어 쓰면 TTC가 엉뚱하게 나옴
    if (!isAnalysisTarget) {
        resetTrackHistory(history);

        return RiskResult{};
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

    // LEAD가 아니면 수치만 채워서 돌려주고 단계 판정은 여기서 끝
    // stabilizeLevel을 돌려버리면 LEAD 아닌 동안 내부 단계가 몰래 올라가 있다가
    // LEAD로 바뀌는 순간 승격 카운트 없이 경고가 튀어나옴
    if (!isLeadTarget) {
        return result;
    }

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