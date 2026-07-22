#include "RiskAnalyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

RiskAnalyzer::RiskAnalyzer(
    double fps,
    int historySize,
    int staleFrameLimit
)
    : fps_(fps),

      // TTC 추세 계산하려면 샘플이 최소한 좀 있어야 해서 8 밑으로는 못 내려가게 막음
      historySize_(
          static_cast<std::size_t>(
              std::max(historySize, 8)
          )
      ),

      staleFrameLimit_(
          std::max(staleFrameLimit, 1)
      ),

      // 같은 ID가 다시 나타나도 공백이 5프레임 넘으면 이전 움직임이랑 안 이어붙임
      // (오래 놓쳤다가 다시 잡히면 박스 크기가 확 달라져서 TTC가 이상하게 튐)
      maximumHistoryGapFrames_(5) {
    if (fps_ <= 0.0) {
        throw std::invalid_argument(
            "RiskAnalyzer FPS must be greater than zero."
        );
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
    const int firstFrame =
        samples.front().frameIndex;

    double meanTime = 0.0;
    double meanValue = 0.0;

    for (const Sample& sample : samples) {
        const double timeSeconds =
            static_cast<double>(
                sample.frameIndex - firstFrame
            ) /
            fps;

        // 박스 높이 그대로가 아니라 log(높이)를 쓰는 이유:
        // 전방 물체의 화면상 높이 h는 대략 실제 거리 Z에 반비례함 (h ≈ k / Z)
        // 그럼 log(h) ≈ log(k) - log(Z)라서, log(h)의 변화율이 거리 감소 추세를 더 잘 보여줌
        // boxHeight가 0 이하로 안 내려가게 최소 1로 clamp
        const double value =
            useLogHeight
                ? std::log(
                      std::max(
                          static_cast<double>(
                              sample.boxHeight
                          ),
                          1.0
                      )
                  )
                : static_cast<double>(
                      sample.groundY
                  );

        meanTime += timeSeconds;
        meanValue += value;
    }

    meanTime /=
        static_cast<double>(
            samples.size()
        );

    meanValue /=
        static_cast<double>(
            samples.size()
        );

    // 최소제곱법 기울기: Σ((t-t̄)(v-v̄)) / Σ((t-t̄)^2)
    double numerator = 0.0;
    double denominator = 0.0;

    for (const Sample& sample : samples) {
        const double timeSeconds =
            static_cast<double>(
                sample.frameIndex - firstFrame
            ) /
            fps;

        const double value =
            useLogHeight
                ? std::log(
                      std::max(
                          static_cast<double>(
                              sample.boxHeight
                          ),
                          1.0
                      )
                  )
                : static_cast<double>(
                      sample.groundY
                  );

        const double timeDifference =
            timeSeconds - meanTime;

        numerator +=
            timeDifference *
            (value - meanValue);

        denominator +=
            timeDifference *
            timeDifference;
    }

    // 샘플들 시간이 사실상 다 같으면 분모가 0에 가까워짐 -> 나눗셈 사고 방지
    if (denominator <= 1.0e-9) {
        return 0.0F;
    }

    return static_cast<float>(
        numerator / denominator
    );
}

// samples[start, start+count) 구간 박스 높이 평균
// 과거 3개 vs 최근 3개 평균 비교해서 heightGrowthRatio 낼 때 씀 (순간 흔들림 완화용)
float RiskAnalyzer::calculateAverageHeight(
    const std::deque<Sample>& samples,
    std::size_t start,
    std::size_t count
) {
    float total = 0.0F;

    for (
        std::size_t index = 0;
        index < count;
        ++index
    ) {
        total +=
            samples[start + index]
                .boxHeight;
    }

    return total /
           static_cast<float>(count);
}

// 트랙 기록 리셋. lead 대상에서 빠졌거나, 공백이 너무 길었거나,
// 과거 기록이랑 지금 기록을 이어 붙이면 안 되는 상황에서 호출됨
// lastSeenFrame은 정리(removeStaleTracks) 판단에 필요해서 여기선 안 건드림
void RiskAnalyzer::resetTrackHistory(
    TrackHistory& history
) {
    history.samples.clear();

    history.stableLevel =
        RiskLevel::Safe;

    history.pendingLevel =
        RiskLevel::Safe;

    history.pendingFrames = 0;
    history.lowerLevelFrames = 0;
}

// rawLevel을 그대로 안 쓰고 몇 프레임 지켜본 다음 반영함
// 올라갈 때: SAFE->CAUTION은 4프레임, (SAFE/CAUTION)->DANGER는 2프레임 연속 확인
// 내려갈 때: DANGER였으면 10프레임, CAUTION이었으면 6프레임 동안 낮은 값이 유지돼야 내려감
// 이렇게 안 하면 박스가 한 프레임 흔들릴 때마다 경고가 깜빡깜빡함
RiskLevel RiskAnalyzer::stabilizeLevel(
    TrackHistory& history,
    RiskLevel rawLevel
) const {
    // 이미 같은 단계면 딱히 승격/강등 처리할 게 없음
    if (rawLevel == history.stableLevel) {
        history.pendingLevel =
            rawLevel;

        history.pendingFrames = 0;
        history.lowerLevelFrames = 0;

        return history.stableLevel;
    }

    // Safe=0, Caution=1, Danger=2 라서 정수 비교로 고저 판단
    const int rawValue =
        static_cast<int>(rawLevel);

    const int stableValue =
        static_cast<int>(
            history.stableLevel
        );

    // 현재보다 높은 값이면 승격 후보
    if (rawValue > stableValue) {
        history.lowerLevelFrames = 0;

        if (
            history.pendingLevel ==
            rawLevel
        ) {
            // 이전이랑 같은 후보가 계속 나오면 카운트 증가
            ++history.pendingFrames;
        } else {
            // 다른 후보가 나오면 새로 잡고 카운트 1부터 시작
            history.pendingLevel =
                rawLevel;

            history.pendingFrames = 1;
        }

        // DANGER는 빨리 반응해야 하니 2프레임, CAUTION은 오경고 줄이려고 4프레임
        const int requiredFrames =
            rawLevel == RiskLevel::Danger
                ? 2
                : 4;

        if (
            history.pendingFrames >=
            requiredFrames
        ) {
            history.stableLevel =
                rawLevel;

            history.pendingFrames = 0;
        }

        return history.stableLevel;
    }

    // 여기 왔으면 rawLevel이 현재보다 낮은 경우
    // 한 프레임 좋아졌다고 바로 경고 꺼버리지 않고 낮은 값이 지속되는지 확인
    history.pendingFrames = 0;
    ++history.lowerLevelFrames;

    const int requiredFrames =
        history.stableLevel ==
                RiskLevel::Danger
            ? 10
            : 6;

    if (
        history.lowerLevelFrames >=
        requiredFrames
    ) {
        history.stableLevel =
            rawLevel;

        history.pendingLevel =
            rawLevel;

        history.lowerLevelFrames = 0;
    }

    return history.stableLevel;
}

RiskResult RiskAnalyzer::update(
    const TrackedObject& trackedObject,
    bool isLeadTarget,
    int currentFrame
) {
    // trackId로 기록 꺼내옴, 처음 보는 ID면 기본값으로 새로 생김
    TrackHistory& history =
        histories_[trackedObject.trackId];

    // 같은 ID라도 검출 공백이 너무 길었으면 예전 움직임이랑 지금 움직임을 안 이어붙임
    if (
        history.lastSeenFrame >= 0 &&
        currentFrame -
                history.lastSeenFrame >
            maximumHistoryGapFrames_
    ) {
        resetTrackHistory(history);
    }

    history.lastSeenFrame =
        currentFrame;

    // lead 차량이 아니면 분석 대상 아님
    // 예전에 lead였던 기록이 남아있으면 나중에 다시 선택될 때 옛날 움직임이랑 섞일 수 있어서
    // lead에서 빠지는 순간 바로 리셋함
    if (!isLeadTarget) {
        resetTrackHistory(history);

        return RiskResult{};
    }

    // log(높이) 계산할 때 0 이하 들어가면 안 되니 최소 1픽셀로 clamp
    const float boxHeight =
        static_cast<float>(
            std::max(
                trackedObject.box.height,
                1
            )
        );

    // 박스 하단 y좌표. 차량이 화면 아래로 내려올수록 커짐
    const float groundY =
        static_cast<float>(
            trackedObject.box.y +
            trackedObject.box.height
        );

    // update()가 같은 프레임에 두 번 불리는 경우 대비 - 중복 추가 말고 덮어씀
    if (
        !history.samples.empty() &&
        history.samples.back()
                .frameIndex ==
            currentFrame
    ) {
        history.samples.back() = {
            currentFrame,
            boxHeight,
            groundY
        };
    } else {
        history.samples.push_back({
            currentFrame,
            boxHeight,
            groundY
        });
    }

    // 최대 보관 개수 넘으면 오래된 것부터 버림
    while (
        history.samples.size() >
        historySize_
    ) {
        history.samples.pop_front();
    }

    RiskResult result;

    result.sampleCount =
        static_cast<int>(
            history.samples.size()
        );

    // 새 단계가 아직 확정 안 됐어도 일단 지금 안정화된 단계를 기본값으로
    result.level =
        history.stableLevel;

    // 샘플이 8개 미만이면 박스 흔들림인지 진짜 확대 추세인지 구분 안 되니까
    // 이 구간에서는 TTC 계산 없이 SAFE 유지
    if (history.samples.size() < 8) {
        return result;
    }

    // 과거/최근 각각 최대 3개 샘플로 평균 냄
    const std::size_t averageCount =
        std::min<std::size_t>(
            3,
            history.samples.size() / 2
        );

    const float firstHeight =
        calculateAverageHeight(
            history.samples,
            0,
            averageCount
        );

    const float lastHeight =
        calculateAverageHeight(
            history.samples,
            history.samples.size() -
                averageCount,
            averageCount
        );

    // 과거 평균 대비 최근 평균 비율. 1.06이면 최근에 6% 커졌다는 뜻
    const float heightGrowthRatio =
        lastHeight /
        std::max(
            firstHeight,
            1.0F
        );

    // log(박스 높이)의 초당 증가율. 클수록 빠르게 커지는 중
    const float logHeightRate =
        calculateRegressionSlope(
            history.samples,
            fps_,
            true
        );

    // 박스 하단 y좌표의 초당 이동속도. 양수면 화면 아래로 이동중
    const float groundSpeed =
        calculateRegressionSlope(
            history.samples,
            fps_,
            false
        );

    // 접근 안 하거나 계산 불가하면 무한대 유지
    float ttcSeconds =
        std::numeric_limits<float>
            ::infinity();

    // 박스 높이가 거의 안 변하는 구간에서 노이즈성 TTC 튀는 거 막으려고
    // logHeightRate가 0.02보다 클 때만 계산함
    // 전방 물체 화면상 높이가 실제 거리랑 반비례한다고 가정하면
    // d(log(height))/dt ≈ 1/TTC 이므로 TTC-P ≈ 1/logHeightRate
    // (카메라 보정된 실거리 기반 정식 TTC 아니라 어디까지나 근사치)
    if (logHeightRate > 0.02F) {
        ttcSeconds =
            1.0F /
            logHeightRate;
    }

    // 너무 작은(멀리 있는) 객체의 불안정한 변화율을 위험으로 오판하지 않기 위한
    // 최소 크기 조건에 쓸 현재 프레임 박스 높이
    const float currentHeight =
        history.samples.back()
            .boxHeight;

    // 아직 안정화 전, 이번 프레임만 놓고 본 위험 단계
    RiskLevel rawLevel =
        RiskLevel::Safe;

    // DANGER 조건: TTC 계산 가능 + 박스 18px 이상 + 최근 높이 6% 이상 증가
    // + 접지점 초당 8px 이상 하강 + TTC 2.5초 이하
    // 여러 조건을 같이 요구하는 이유는 박스 하나만 순간적으로 커지는 오검출 걸러내려고
    if (
        std::isfinite(ttcSeconds) &&
        currentHeight >= 18.0F &&
        heightGrowthRatio >= 1.06F &&
        groundSpeed >= 8.0F &&
        ttcSeconds <= 2.5F
    ) {
        rawLevel =
            RiskLevel::Danger;
    }
    // CAUTION은 DANGER보다 완화된 기준 (12px / 2.5% / 2.5px/s / 5초)
    else if (
        std::isfinite(ttcSeconds) &&
        currentHeight >= 12.0F &&
        heightGrowthRatio >= 1.025F &&
        groundSpeed >= 2.5F &&
        ttcSeconds <= 5.0F
    ) {
        rawLevel =
            RiskLevel::Caution;
    }

    // 이번 프레임 rawLevel 그대로 안 쓰고 안정화된 값으로 반환
    result.level =
        stabilizeLevel(
            history,
            rawLevel
        );

    // 샘플 충분하고 TTC가 유한할 때만 valid=true
    // (샘플은 충분한데 차가 안 가까워지면 TTC는 무한대라 화면엔 "--"로 표시됨)
    result.valid =
        std::isfinite(ttcSeconds);

    result.ttcSeconds =
        ttcSeconds;

    result.logHeightRatePerSecond =
        logHeightRate;

    result.groundSpeedPixelsPerSecond =
        groundSpeed;

    result.heightGrowthRatio =
        heightGrowthRatio;

    return result;
}

// 오래 안 보인 트랙 기록 삭제
// update()가 안 불리는 객체는 lastSeenFrame이 안 갱신되니까
// currentFrame과의 차이가 staleFrameLimit_ 넘으면 삭제 대상
void RiskAnalyzer::removeStaleTracks(
    int currentFrame
) {
    // 순회하면서 바로 erase하면 iterator 깨지니까 삭제할 것만 먼저 모아둠
    std::vector<int> staleIds;

    for (const auto& entry : histories_) {
        const int trackId =
            entry.first;

        const TrackHistory& history =
            entry.second;

        if (
            history.lastSeenFrame >= 0 &&
            currentFrame -
                    history.lastSeenFrame >
                staleFrameLimit_
        ) {
            staleIds.push_back(
                trackId
            );
        }
    }

    for (const int trackId : staleIds) {
        histories_.erase(trackId);
    }
}

std::string RiskAnalyzer::toString(
    RiskLevel level
) {
    switch (level) {
        case RiskLevel::Safe:
            return "SAFE";

        case RiskLevel::Caution:
            return "CAUTION";

        case RiskLevel::Danger:
            return "DANGER";
    }

    // 여기 오면 안 되지만 혹시 모르니 기본값
    return "UNKNOWN";
}