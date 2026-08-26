#pragma once

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>

/*
 * captureTimestampNs가 어느 클럭에서 나온 값인지
 *
 * - VideoPts       : 영상 파일의 PTS (스트림 시작 기준 상대 시간)
 *                    실제 센서 시각이 아니므로 steady_clock과 직접 비교할 수 없음
 * - V4l2Monotonic  : V4L2 버퍼의 CLOCK_MONOTONIC timestamp
 *                    Linux에서 std::chrono::steady_clock과 같은 클럭이므로
 *                    decision_ts와 빼서 end-to-end 지연을 계산할 수 있음
 */
enum class CaptureTimestampSource {
    VideoPts,
    V4l2Monotonic
};

inline std::string toString(CaptureTimestampSource source) {
    switch (source) {
        case CaptureTimestampSource::VideoPts: return "video_pts";
        case CaptureTimestampSource::V4l2Monotonic: return "v4l2_monotonic";
    }
    return "unknown";
}

/*
 * FrameSource가 반환하는 프레임 하나와 캡처 메타데이터
 *
 * - image              : BGR 프레임
 * - frameSeq           : 소스 내 프레임 순번, 첫 프레임이 0
 *                        V4L2의 v4l2_buffer.sequence와 같은 규칙이라
 *                        카메라에서 프레임이 버려지면 번호가 건너뛴다
 * - captureTimestampNs : 캡처 시각(ns). 단위를 ns로 둔 이유는
 *                        V4L2 timeval(us), steady_clock time_since_epoch(ns),
 *                        ROS2 rclcpp::Time(ns)이 모두 손실 없이 int64 ns로 바뀌기 때문
 * - captureTimestampSource : 위 timestamp의 클럭 도메인
 */
struct Frame {
    cv::Mat image;
    std::int64_t frameSeq = 0;
    std::int64_t captureTimestampNs = 0;
    CaptureTimestampSource captureTimestampSource = CaptureTimestampSource::VideoPts;
};
