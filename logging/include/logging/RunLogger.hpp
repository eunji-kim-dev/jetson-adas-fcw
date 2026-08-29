#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>

/*
 * 실행 단위 조건 — run_summary.json 에 기록
 *
 * git_commit / git_dirty / build_type / compiler 는 빌드 시점에 생성된
 * BuildInfo.hpp 에서 RunLogger 가 직접 채우므로 여기 없음.
 */
struct RunMetadata {
    std::string runId;
    std::string hardware;          // 비어 있으면 RunLogger 가 /proc 에서 자동 감지
    std::string powerMode;
    std::string backend;
    std::string precision;

    std::optional<double> temperatureStartC;  // 시작 시 SoC 온도(°C), x86은 값 없음
    std::optional<bool> jetsonClocks;         // jetson_clocks 적용 여부, x86은 값 없음

    std::string opencvVersion;

    std::string inputSource;       // "video_file" / "camera"
    std::string inputName;
    std::string inputHash;         // hashFile() 결과, 비어 있어도 됨
    std::string resolution;        // "1280x720"
    double sourceFps = 0.0;
    std::string captureTsClock;    // "stream" / "monotonic" / "realtime"
    std::string captureTsSource;   // "video_pts" / "v4l2_monotonic"

    std::string model;
    std::string modelHash;
    std::string fullCropStrategy;  // "full+crop_every_frame"
    int detectionInterval = 1;
    double confidenceThreshold = 0.0;
    double nmsThreshold = 0.0;

    int warmupFrames = 0;
    int measuredFrames = 0;        // 0 = 끝까지
    double deadlineMs = 0.0;
};

/*
 * 프레임 단위 기록 — raw_frame_log.csv 의 한 행
 *
 * 값이 없는 측정 항목은 std::optional 을 비워 두면 빈 칸으로 기록된다.
 * 시각(*TsNs)은 모두 int64 ns. dequeue/decision 은 steady_clock,
 * capture 는 Frame 이 준 값(clock 은 captureTsClock 참조).
 */
struct FrameRecord {
    std::int64_t frame = 0;                // 앱이 처리한 순번 (1부터)
    std::int64_t frameSeq = 0;             // 소스의 프레임 번호 (0부터)
    std::int64_t captureTsNs = 0;
    std::string captureTsClock;
    std::string captureTsSource;
    std::int64_t dequeueTsNs = 0;
    std::int64_t decisionTsNs = 0;

    double captureMs = 0.0;
    std::optional<double> sceneMs;
    double preprocessFullMs = 0.0;
    double inferenceFullMs = 0.0;
    double postprocessFullMs = 0.0;
    double preprocessCropMs = 0.0;
    double inferenceCropMs = 0.0;
    double postprocessCropMs = 0.0;
    double mergeMs = 0.0;
    double detectMs = 0.0;
    double trackingMs = 0.0;
    std::optional<double> decisionMs;
    double totalProcessingMs = 0.0;        // decision_ts - dequeue_ts
    double outputMs = 0.0;
    bool deadlineMiss = false;

    int detections = 0;
    int tracks = 0;
    std::optional<int> leadId;             // adas: 없으면 -1, perception_demo: 빈 칸
    std::optional<bool> leadFound;
    std::optional<double> ttcP;            // 초, 유효하지 않으면 빈 칸
    std::optional<std::string> riskState;
    std::optional<std::string> warningState;
    std::optional<bool> sceneChanged;
};

/*
 * 실행 1회의 로그 두 파일을 쓴다
 *   <runsRoot>/<runId>/raw_frame_log.csv
 *   <runsRoot>/<runId>/run_summary.json
 *
 * 생성 시점에 run_summary.json 을 조건만으로 먼저 쓰고,
 * finish() 에서 frames_processed / elapsed_s 를 채워 다시 쓴다.
 */
class RunLogger {
public:
    static constexpr int kSchemaVersion = 2;

    RunLogger(const std::string& runsRoot, RunMetadata metadata);
    ~RunLogger();

    void writeFrame(const FrameRecord& record);

    void setTemperatureEnd(std::optional<double> celsius) {
    temperatureEndC_ = celsius;
    }

    void finish();

    const std::string& runDirectory() const { return runDirectory_; }
    const std::string& gitCommit() const { return gitCommit_; }
    bool gitDirty() const { return gitDirty_; }

    // 도우미
    static std::int64_t monotonicNowNs();
    static std::string hashFile(const std::string& path);   // "fnv1a64:<hex>", 실패 시 ""
    static std::string detectHardware();
    static std::string defaultRunId(const std::string& inputStem, const std::string& backend);

private:
    void writeSummary();

    RunMetadata metadata_;
    std::string runDirectory_;
    std::string gitCommit_;
    bool gitDirty_;
    std::string buildType_;
    std::string compiler_;
    std::string startedAtUtc_;
    std::int64_t monotonicAtStartNs_;
    std::ofstream frameLog_;
    std::int64_t framesProcessed_;
    double elapsedSeconds_;
    std::optional<double> temperatureEndC_;
    bool finished_;
};
