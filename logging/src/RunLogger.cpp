#include "logging/RunLogger.hpp"

#include "BuildInfo.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// raw_frame_log.csv 헤더 — FrameRecord 필드 순서와 일치해야 함
const char* const kCsvHeader =
    "run_id,frame,frame_seq,"
    "capture_ts_ns,capture_ts_clock,capture_ts_source,dequeue_ts_ns,decision_ts_ns,"
    "capture_ms,scene_ms,"
    "preprocess_full_ms,inference_full_ms,postprocess_full_ms,"
    "preprocess_crop_ms,inference_crop_ms,postprocess_crop_ms,"
    "merge_ms,detect_ms,tracking_ms,decision_ms,total_processing_ms,output_ms,deadline_miss,"
    "detections,tracks,lead_id,lead_found,ttc_p,risk_state,warning_state,scene_changed";

std::string jsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

std::string jsonString(const std::string& text) { return "\"" + jsonEscape(text) + "\""; }

std::string formatDouble(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string nowUtcIso8601() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

std::string nowLocalCompact() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
    localtime_r(&now, &local);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d-%H%M%S", &local);
    return buffer;
}

std::string trim(std::string text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\0')) text.pop_back();
    std::size_t start = 0;
    while (start < text.size() && text[start] == ' ') ++start;
    return text.substr(start);
}

// CSV 셀: optional 이 비어 있으면 빈 칸
std::string cell(const std::optional<double>& value) { return value ? formatDouble(*value, 3) : ""; }
std::string cell(const std::optional<int>& value) { return value ? std::to_string(*value) : ""; }
std::string cell(const std::optional<bool>& value) { return value ? (*value ? "1" : "0") : ""; }
std::string cell(const std::optional<std::string>& value) { return value ? *value : ""; }

} // namespace

RunLogger::RunLogger(const std::string& runsRoot, RunMetadata metadata)
    : metadata_(std::move(metadata)),
      gitCommit_(BUILD_INFO_GIT_COMMIT),
      gitDirty_(BUILD_INFO_GIT_DIRTY != 0),
      buildType_(BUILD_INFO_BUILD_TYPE),
      compiler_(BUILD_INFO_COMPILER),
      startedAtUtc_(nowUtcIso8601()),
      monotonicAtStartNs_(monotonicNowNs()),
      framesProcessed_(0),
      elapsedSeconds_(0.0),
      finished_(false) {
    if (metadata_.hardware.empty()) metadata_.hardware = detectHardware();

    runDirectory_ = runsRoot + "/" + metadata_.runId;
    std::filesystem::create_directories(runDirectory_);

    frameLog_.open(runDirectory_ + "/raw_frame_log.csv");
    if (!frameLog_.is_open()) throw std::runtime_error("raw_frame_log.csv 생성 실패: " + runDirectory_);
    frameLog_ << kCsvHeader << '\n';

    writeSummary();
}

RunLogger::~RunLogger() {
    if (!finished_) finish();
}

void RunLogger::writeFrame(const FrameRecord& r) {
    frameLog_ << metadata_.runId << ',' << r.frame << ',' << r.frameSeq << ','
              << r.captureTsNs << ',' << r.captureTsClock << ',' << r.captureTsSource << ','
              << r.dequeueTsNs << ',' << r.decisionTsNs << ','
              << formatDouble(r.captureMs, 3) << ',' << cell(r.sceneMs) << ','
              << formatDouble(r.preprocessFullMs, 3) << ',' << formatDouble(r.inferenceFullMs, 3) << ',' << formatDouble(r.postprocessFullMs, 3) << ','
              << formatDouble(r.preprocessCropMs, 3) << ',' << formatDouble(r.inferenceCropMs, 3) << ',' << formatDouble(r.postprocessCropMs, 3) << ','
              << formatDouble(r.mergeMs, 3) << ',' << formatDouble(r.detectMs, 3) << ',' << formatDouble(r.trackingMs, 3) << ','
              << cell(r.decisionMs) << ',' << formatDouble(r.totalProcessingMs, 3) << ',' << formatDouble(r.outputMs, 3) << ','
              << (r.deadlineMiss ? 1 : 0) << ','
              << r.detections << ',' << r.tracks << ','
              << cell(r.leadId) << ',' << cell(r.leadFound) << ',' << cell(r.ttcP) << ','
              << cell(r.riskState) << ',' << cell(r.warningState) << ',' << cell(r.sceneChanged)
              << '\n';
    ++framesProcessed_;
}

void RunLogger::finish() {
    if (finished_) return;
    finished_ = true;
    elapsedSeconds_ = static_cast<double>(monotonicNowNs() - monotonicAtStartNs_) / 1.0e9;
    frameLog_.close();
    writeSummary();
}

void RunLogger::writeSummary() {
    std::ofstream json(runDirectory_ + "/run_summary.json");
    if (!json.is_open()) throw std::runtime_error("run_summary.json 생성 실패: " + runDirectory_);

    const RunMetadata& m = metadata_;
    json << "{\n"
         << "  \"schema_version\": " << kSchemaVersion << ",\n"
         << "  \"run_id\": " << jsonString(m.runId) << ",\n"
         << "  \"started_at_utc\": " << jsonString(startedAtUtc_) << ",\n"
         << "  \"monotonic_at_start_ns\": " << monotonicAtStartNs_ << ",\n"
         << "\n"
         << "  \"git_commit\": " << jsonString(gitCommit_) << ",\n"
         << "  \"git_dirty\": " << (gitDirty_ ? "true" : "false") << ",\n"
         << "  \"build_type\": " << jsonString(buildType_) << ",\n"
         << "  \"compiler\": " << jsonString(compiler_) << ",\n"
         << "  \"opencv_version\": " << jsonString(m.opencvVersion) << ",\n"
         << "\n"
         << "  \"hardware\": " << jsonString(m.hardware) << ",\n"
         << "  \"power_mode\": " << jsonString(m.powerMode) << ",\n"
         << "  \"backend\": " << jsonString(m.backend) << ",\n"
         << "  \"precision\": " << jsonString(m.precision) << ",\n"
         << "\n"
         << "  \"input_source\": " << jsonString(m.inputSource) << ",\n"
         << "  \"input_name\": " << jsonString(m.inputName) << ",\n"
         << "  \"input_hash\": " << jsonString(m.inputHash) << ",\n"
         << "  \"resolution\": " << jsonString(m.resolution) << ",\n"
         << "  \"source_fps\": " << formatDouble(m.sourceFps, 3) << ",\n"
         << "  \"capture_ts_clock\": " << jsonString(m.captureTsClock) << ",\n"
         << "  \"capture_ts_source\": " << jsonString(m.captureTsSource) << ",\n"
         << "\n"
         << "  \"model\": " << jsonString(m.model) << ",\n"
         << "  \"model_hash\": " << jsonString(m.modelHash) << ",\n"
         << "  \"full_crop_strategy\": " << jsonString(m.fullCropStrategy) << ",\n"
         << "  \"detection_interval\": " << m.detectionInterval << ",\n"
         << "  \"confidence_threshold\": " << formatDouble(m.confidenceThreshold, 3) << ",\n"
         << "  \"nms_threshold\": " << formatDouble(m.nmsThreshold, 3) << ",\n"
         << "\n"
         << "  \"warmup_frames\": " << m.warmupFrames << ",\n"
         << "  \"measured_frames\": " << m.measuredFrames << ",\n"
         << "  \"deadline_ms\": " << formatDouble(m.deadlineMs, 3) << ",\n"
         << "\n"
         << "  \"frames_processed\": " << framesProcessed_ << ",\n"
         << "  \"elapsed_s\": " << formatDouble(elapsedSeconds_, 3) << "\n"
         << "}\n";
}

std::int64_t RunLogger::monotonicNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// FNV-1a 64bit — 의존성 없이 "어느 파일이었는가"를 식별하는 용도 (보안 해시 아님)
std::string RunLogger::hashFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return "";

    std::uint64_t hash = 14695981039346656037ULL;
    std::vector<char> buffer(1 << 20);
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = file.gcount();
        for (std::streamsize i = 0; i < count; ++i) {
            hash ^= static_cast<unsigned char>(buffer[static_cast<std::size_t>(i)]);
            hash *= 1099511628211ULL;
        }
    }

    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

// Jetson: /proc/device-tree/model ("NVIDIA Jetson Orin Nano ...")
// x86   : /proc/cpuinfo 의 model name
std::string RunLogger::detectHardware() {
    {
        std::ifstream model("/proc/device-tree/model");
        if (model.is_open()) {
            std::string line;
            std::getline(model, line);
            line = trim(line);
            if (!line.empty()) return line;
        }
    }
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("model name", 0) == 0) {
                const std::size_t colon = line.find(':');
                if (colon != std::string::npos) return trim(line.substr(colon + 1));
            }
        }
    }
    return "unknown";
}

std::string RunLogger::defaultRunId(const std::string& inputStem, const std::string& backend) {
    return nowLocalCompact() + "_" + inputStem + "_" + backend;
}
