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
    std::string captureMode;       // "sync" (읽고 처리를 한 스레드에서) / "threaded" (캡처 스레드 분리, 최신 프레임 우선)

    std::string model;
    std::string modelHash;
    std::string fullCropStrategy;  // "full+crop_every_frame"
    std::string cropModel;         // crop 추론 모델 경로. 전체 프레임과 같은 모델이면 "same_as_model"
    std::string cropInput;         // crop 추론 입력 크기 "HxW". 기본 "640x640"
    std::string laneRoi;           // "default" 또는 "x1,y1,...,x4,y4" (--lane-roi 픽셀 좌표)
    std::string leadRule;          // "none" 또는 "overlap,gap" 처럼 --lead-rule 목록 (adas). perception_demo 는 비움
    // TTR-H 실험 (adas --ttc-mode). perception_demo 는 비움. proxy 면 노면 값은 빈 칸·null
    std::string ttcMode;                       // "proxy" / "homography" / "both"
    std::string roadPoints;                    // "x1,y1,...,x4,y4" (노면 4점)
    std::optional<double> roadWidthM;          // 노면 사각형 폭 (m)
    std::optional<double> roadLengthM;         // 노면 사각형 길이 (m)
    std::optional<int> roadFrame;              // 노면 4점을 찍은 프레임
    std::string roadResolution;                // 노면 4점을 찍은 해상도 "1920x1080"
    std::optional<int> bonnetY;                // 보닛 선 y. null 이면 화면 맨 아래 줄 기준
    std::string roadStatus;                    // "assumed" / "verified"
    std::optional<double> d0M;                 // 예약 값 (계산에 안 씀)
    std::optional<double> referenceDepthM;     // Z_c: 화면 가운데·보닛 선(없으면 맨 아래 줄)의 깊이 (m)
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

    std::int64_t sourceDrops = 0;          // 직전 구간에서 소스(카메라) 쪽이 놓친 프레임 수. 파일은 0
    std::int64_t appDrops = 0;             // 직전 구간에서 프로그램이 버린 프레임 수 (threaded 모드). sync 는 0

    // TTR-H (adas --ttc-mode homography|both 일 때만, LEAD 기준). 값이 없으면 빈 칸
    std::optional<double> distanceM;       // LEAD 접지점 깊이 Z_now (m). 접지점이 무효면 빈 칸
    std::optional<double> ttrH;            // TTR-H 초 (게이트 뒤). 유효하지 않으면 빈 칸
    std::optional<bool> hValid;            // TTR-H 계산 여부
    std::optional<std::string> hLevel;     // H 단계 (게이트 뒤)
    std::optional<std::string> pLevel;     // P 단계 (게이트 뒤, risk_state 와 같은 값). LEAD 없으면 빈 칸
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
    static constexpr int kSchemaVersion = 7;   // v7: ttc_mode·노면 4점 값·z_c_m 추가, raw_frame_log 에 distance_m·ttr_h·h_valid·h_level·p_level 열 추가. v6: lead_rule 추가. v5: capture_mode 추가, raw_frame_log 에 source_drops·app_drops 열 추가

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
