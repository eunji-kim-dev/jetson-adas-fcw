#include "perception/VideoFileSource.hpp"

#include <opencv2/videoio.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

VideoFileSource::VideoFileSource(const std::string& path)
    : capture_(std::make_unique<cv::VideoCapture>(path)), width_(0), height_(0), fps_(0.0), nextFrameSeq_(0) {
    if (!capture_->isOpened()) throw std::runtime_error("영상 열기 실패: " + path);

    width_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = capture_->get(cv::CAP_PROP_FPS);
    if (fps_ <= 0.0) fps_ = 30.0;
}

// unique_ptr<cv::VideoCapture>의 소멸자는 완전한 타입이 필요하므로 여기서 정의
VideoFileSource::~VideoFileSource() = default;

bool VideoFileSource::read(Frame& frame) {
    if (!capture_->read(frame.image)) return false;

    frame.frameSeq = nextFrameSeq_++;

    // CAP_PROP_POS_MSEC는 read() 직후 "방금 디코드한 프레임"의 PTS를 ms로 돌려준다
    // (FFmpeg 백엔드 기준, 4.x에서 실측 확인)
    // 첫 프레임은 정상적으로 0이므로, 두 번째 프레임부터도 0 이하가 나오면
    // 이 백엔드는 PTS를 지원하지 않는 것으로 보고 frameSeq / fps 로 대체한다
    const double positionMilliseconds = capture_->get(cv::CAP_PROP_POS_MSEC);
    const bool ptsValid = positionMilliseconds > 0.0 || (frame.frameSeq == 0 && positionMilliseconds == 0.0);

    if (ptsValid) {
        frame.captureTimestampNs = static_cast<std::int64_t>(std::llround(positionMilliseconds * 1.0e6));
    } else {
        frame.captureTimestampNs = static_cast<std::int64_t>(std::llround(static_cast<double>(frame.frameSeq) * 1.0e9 / fps_));
    }
    frame.captureTimestampSource = CaptureTimestampSource::VideoPts;

    return true;
}

int VideoFileSource::width() const { return width_; }

int VideoFileSource::height() const { return height_; }

double VideoFileSource::fps() const { return fps_; }
