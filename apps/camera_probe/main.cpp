/*
 * 카메라 입력 경로만 확인하는 도구 (추론 없음)
 *
 * V4l2CameraSource 로 프레임을 받아 Frame Age(받은 직후 시각 - 노출 시작 시각)와 Frame Drop 만 잼
 * 추론이 없을 때의 입력 지연 하한임. adas 의 Frame Age 에서 이 값을 빼면 처리 때문에 늘어난 몫이 나옴
 *
 * 사용법: camera_probe [/dev/video0] [--frames N] [--warmup-frames N] [--threaded] [--busy-ms MS] [--csv PATH]
 *   --frames N          받을 프레임 수 (기본 900 = 30fps 30초)
 *   --warmup-frames N   앞 N 프레임은 요약에서 뺌 (기본 40. 시작 직후 약 1.3초 멈춤 구간)
 *   --threaded          ThreadedFrameSource 를 씌움 (최신 프레임 우선)
 *   --busy-ms MS        프레임마다 MS 밀리초 동안 바쁘게 기다려 처리 지연을 흉내냄 (기본 0)
 *                       --threaded --busy-ms 50 이면 프로그램 쪽 버림(app drop)이 생겨야 정상
 *   --csv PATH          프레임별 기록 저장 (frame,seq,capture_ts_ns,dequeue_ts_ns,age_ms,source_drops,app_drops)
 */
#include "perception/Frame.hpp"
#include "perception/FrameSource.hpp"
#include "perception/ThreadedFrameSource.hpp"
#include "perception/V4l2CameraSource.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = (static_cast<double>(values.size()) - 1.0) * p / 100.0;
    const std::size_t low = static_cast<std::size_t>(rank);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (rank - static_cast<double>(low));
}

std::int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string device = "/dev/video0";
    int frames = 900;
    int warmupFrames = 40;
    bool threaded = false;
    double busyMs = 0.0;
    std::string csvPath;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[ERROR] " << name << " 옵션에 값이 없음\n";
                std::exit(1);
            }
            return argv[++i];
        };
        if (argument == "--frames") frames = std::stoi(next("--frames"));
        else if (argument == "--warmup-frames") warmupFrames = std::stoi(next("--warmup-frames"));
        else if (argument == "--threaded") threaded = true;
        else if (argument == "--busy-ms") busyMs = std::stod(next("--busy-ms"));
        else if (argument == "--csv") csvPath = next("--csv");
        else if (argument.rfind("--", 0) == 0) {
            std::cerr << "[ERROR] 알 수 없는 옵션: " << argument << '\n';
            return 1;
        } else device = argument;
    }

    std::unique_ptr<FrameSource> source;
    try {
        source = std::make_unique<V4l2CameraSource>(device, 640, 480, 30.0);
        if (threaded) source = std::make_unique<ThreadedFrameSource>(std::move(source));
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
    std::cout << "[INFO] " << device << " " << source->width() << "x" << source->height() << " @ " << source->fps() << "fps"
              << " | 캡처 " << (threaded ? "스레드 분리" : "동기") << " | busy " << busyMs << " ms | " << frames << " 프레임\n";

    std::ofstream csv;
    if (!csvPath.empty()) {
        csv.open(csvPath);
        if (!csv.is_open()) {
            std::cerr << "[ERROR] CSV 생성 실패: " << csvPath << '\n';
            return 1;
        }
        csv << "frame,seq,capture_ts_ns,dequeue_ts_ns,age_ms,source_drops,app_drops\n";
    }

    std::vector<double> ages, intervals;
    std::int64_t sourceDrops = 0, appDrops = 0, previousCaptureNs = 0;
    int received = 0;
    bool monotonicWarned = false;

    Frame frame;
    while (received < frames && source->read(frame)) {
        const std::int64_t dequeueNs = nowNs();
        ++received;

        if (frame.captureTimestampClock != CaptureTimestampClock::Monotonic && !monotonicWarned) {
            std::cerr << "[WARN] capture timestamp 가 monotonic 이 아님 (" << toString(frame.captureTimestampClock) << ") — Frame Age 는 참고값\n";
            monotonicWarned = true;
        }
        const double ageMs = static_cast<double>(dequeueNs - frame.captureTimestampNs) / 1.0e6;

        if (csv.is_open()) {
            csv << received << ',' << frame.frameSeq << ',' << frame.captureTimestampNs << ',' << dequeueNs << ','
                << std::fixed << std::setprecision(3) << ageMs << ',' << frame.droppedBySource << ',' << frame.droppedByApp << '\n';
        }
        if (received % 30 == 0) {
            std::cout << std::fixed << std::setprecision(1) << "[CAM] frame=" << received << " seq=" << frame.frameSeq
                      << " age=" << ageMs << " ms | drop src=" << frame.droppedBySource << " app=" << frame.droppedByApp << '\n';
        }

        // 준비 구간을 뺀 요약
        if (received > warmupFrames) {
            ages.push_back(ageMs);
            sourceDrops += frame.droppedBySource;
            appDrops += frame.droppedByApp;
            if (previousCaptureNs > 0) intervals.push_back(static_cast<double>(frame.captureTimestampNs - previousCaptureNs) / 1.0e6);
        }
        previousCaptureNs = frame.captureTimestampNs;

        // 처리 지연 흉내: sleep 이 아니라 바쁘게 기다려 CPU 를 실제로 씀
        if (busyMs > 0.0) {
            const std::int64_t until = nowNs() + static_cast<std::int64_t>(busyMs * 1.0e6);
            while (nowNs() < until) {}
        }
    }

    std::cout << "\n받은 프레임: " << received << " (요약은 앞 " << warmupFrames << " 프레임 제외, " << ages.size() << " 프레임)\n";
    std::cout << std::fixed << std::setprecision(1)
              << "Frame Age p50 / p95 / 최대: " << percentile(ages, 50) << " / " << percentile(ages, 95) << " / "
              << (ages.empty() ? 0.0 : *std::max_element(ages.begin(), ages.end())) << " ms\n"
              << "촬영 간격 p50 / 최대: " << percentile(intervals, 50) << " / "
              << (intervals.empty() ? 0.0 : *std::max_element(intervals.begin(), intervals.end())) << " ms\n"
              << "Frame Drop 카메라 쪽 / 프로그램 쪽: " << sourceDrops << " / " << appDrops << '\n';
    if (!csvPath.empty()) std::cout << "CSV: " << csvPath << '\n';
    return 0;
}