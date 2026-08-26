#pragma once

#include "perception/FrameSource.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace cv {
class VideoCapture;
}

/*
 * 영상 파일 입력 (기존 main.cpp의 cv::VideoCapture 사용부를 이동)
 *
 * - frameSeq           : 읽은 순서대로 0부터 증가
 * - captureTimestampNs : 방금 읽은 프레임의 컨테이너 PTS(CAP_PROP_POS_MSEC)를 ns로 변환
 *                        백엔드가 PTS를 주지 않으면 frameSeq / fps 로 계산
 * - captureTimestampSource : 항상 VideoPts
 *
 * cv::VideoCapture는 전방 선언 + unique_ptr로 감춰서
 * 이 헤더를 include하는 쪽이 opencv_videoio에 의존하지 않게 한다.
 */
class VideoFileSource : public FrameSource {
public:
    // 파일을 열지 못하면 std::runtime_error
    explicit VideoFileSource(const std::string& path);
    ~VideoFileSource() override;

    bool read(Frame& frame) override;

    int width() const override;
    int height() const override;
    // 컨테이너가 fps를 알려주지 않으면(0 이하) 30.0으로 간주
    double fps() const override;

private:
    std::unique_ptr<cv::VideoCapture> capture_;
    int width_;
    int height_;
    double fps_;
    std::int64_t nextFrameSeq_;
};
