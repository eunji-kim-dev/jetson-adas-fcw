#include "perception/V4l2CameraSource.hpp"

#include <opencv2/imgproc.hpp>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>

namespace {

// ioctl 이 시그널(EINTR)로 끊기면 다시 시도함
int xioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result == -1 && errno == EINTR);
    return result;
}

std::string errnoText() { return std::string(std::strerror(errno)); }

}  // namespace

V4l2CameraSource::V4l2CameraSource(const std::string& devicePath, int width, int height, double fps, unsigned bufferCount)
    : devicePath_(devicePath), fd_(-1), width_(width), height_(height), bytesPerLine_(0), fps_(fps), streaming_(false), lastSequence_(-1) {
    // O_NONBLOCK: DQBUF 가 멈추지 않게 함. 프레임 대기는 read() 의 poll() 이 맡음
    fd_ = open(devicePath.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) throw std::runtime_error("카메라 열기 실패: " + devicePath + " (" + errnoText() + ")");

    try {
        v4l2_capability capability{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) throw std::runtime_error("VIDIOC_QUERYCAP 실패: " + errnoText());
        // capabilities 는 같은 카메라의 노드 전체(영상 + 메타데이터) 합이라 /dev/video1(메타데이터)도 통과함
        // 이 노드만의 값은 device_caps 에 있음 (V4L2_CAP_DEVICE_CAPS 가 켜져 있을 때)
        const __u32 nodeCapabilities = (capability.capabilities & V4L2_CAP_DEVICE_CAPS) ? capability.device_caps : capability.capabilities;
        if (!(nodeCapabilities & V4L2_CAP_VIDEO_CAPTURE)) throw std::runtime_error("영상 캡처 노드가 아님 (메타데이터 노드?): " + devicePath);
        if (!(nodeCapabilities & V4L2_CAP_STREAMING)) throw std::runtime_error("MMAP 스트리밍을 지원하지 않는 장치임: " + devicePath);

        setFormat(width, height);
        setFrameRate(fps);
        disableAutoExposure();
        mapBuffers(bufferCount);
        startStreaming();
    } catch (...) {
        stopStreaming();
        close(fd_);
        fd_ = -1;
        throw;
    }
}

V4l2CameraSource::~V4l2CameraSource() {
    stopStreaming();
    if (fd_ >= 0) close(fd_);
}

void V4l2CameraSource::setFormat(int width, int height) {
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<__u32>(width);
    format.fmt.pix.height = static_cast<__u32>(height);
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(fd_, VIDIOC_S_FMT, &format) < 0) throw std::runtime_error("VIDIOC_S_FMT 실패: " + errnoText());

    // 드라이버가 요청과 다른 값으로 바꿀 수 있으므로 실제 적용된 값을 씀
    if (format.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) throw std::runtime_error("카메라가 YUYV 포맷을 지원하지 않음: " + devicePath_);
    width_ = static_cast<int>(format.fmt.pix.width);
    height_ = static_cast<int>(format.fmt.pix.height);
    bytesPerLine_ = format.fmt.pix.bytesperline > 0 ? format.fmt.pix.bytesperline : static_cast<std::size_t>(width_) * 2;
}

void V4l2CameraSource::setFrameRate(double fps) {
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1;
    parameters.parm.capture.timeperframe.denominator = static_cast<__u32>(std::lround(fps));
    if (xioctl(fd_, VIDIOC_S_PARM, &parameters) < 0) throw std::runtime_error("VIDIOC_S_PARM(fps) 실패: " + errnoText());

    const v4l2_fract& applied = parameters.parm.capture.timeperframe;
    if (applied.numerator > 0 && applied.denominator > 0) {
        fps_ = static_cast<double>(applied.denominator) / static_cast<double>(applied.numerator);
    }
}

void V4l2CameraSource::disableAutoExposure() {
    // V4L2_EXPOSURE_MANUAL(=1) = v4l2-ctl 의 auto_exposure=1
    v4l2_control control{};
    control.id = V4L2_CID_EXPOSURE_AUTO;
    control.value = V4L2_EXPOSURE_MANUAL;
    if (xioctl(fd_, VIDIOC_S_CTRL, &control) < 0) throw std::runtime_error("자동 노출 끄기(VIDIOC_S_CTRL EXPOSURE_AUTO) 실패: " + errnoText());

    // 실제로 바뀌었는지 되읽음. 자동 노출이 남아 있으면 25fps 로 떨어질 수 있어 측정을 시작하지 않음
    v4l2_control check{};
    check.id = V4L2_CID_EXPOSURE_AUTO;
    if (xioctl(fd_, VIDIOC_G_CTRL, &check) < 0) throw std::runtime_error("자동 노출 값 읽기 실패: " + errnoText());
    if (check.value != V4L2_EXPOSURE_MANUAL) throw std::runtime_error("자동 노출이 꺼지지 않음 (auto_exposure=" + std::to_string(check.value) + ")");
}

void V4l2CameraSource::mapBuffers(unsigned bufferCount) {
    v4l2_requestbuffers request{};
    request.count = bufferCount;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0) throw std::runtime_error("VIDIOC_REQBUFS 실패: " + errnoText());
    if (request.count < 2) throw std::runtime_error("카메라 버퍼가 2개 미만임: " + std::to_string(request.count));

    buffers_.resize(request.count);
    for (unsigned index = 0; index < request.count; ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) throw std::runtime_error("VIDIOC_QUERYBUF 실패: " + errnoText());

        void* start = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
        if (start == MAP_FAILED) throw std::runtime_error("카메라 버퍼 mmap 실패: " + errnoText());
        buffers_[index].start = start;
        buffers_[index].length = buffer.length;
    }
}

void V4l2CameraSource::startStreaming() {
    // 버퍼를 전부 드라이버에 넘기고 스트리밍 시작
    for (unsigned index = 0; index < buffers_.size(); ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) throw std::runtime_error("VIDIOC_QBUF 실패: " + errnoText());
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) throw std::runtime_error("VIDIOC_STREAMON 실패: " + errnoText());
    streaming_ = true;
}

void V4l2CameraSource::stopStreaming() {
    if (streaming_) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }
    for (MappedBuffer& buffer : buffers_) {
        if (buffer.start != nullptr) munmap(buffer.start, buffer.length);
        buffer.start = nullptr;
    }
    buffers_.clear();
}

bool V4l2CameraSource::read(Frame& frame) {
    if (fd_ < 0 || !streaming_) return false;

    // --camera-zero-copy: 지난 프레임 버퍼를 이제 돌려줌. 그 프레임의 GPU 작업(전처리·추론)은 detect() 안의 sync 로 끝났음
    if (heldIndex_ >= 0) {
        v4l2_buffer held{};
        held.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        held.memory = V4L2_MEMORY_MMAP;
        held.index = static_cast<__u32>(heldIndex_);
        heldIndex_ = -1;
        frame.yuyv = YuyvBuffer();   // 돌려준 버퍼를 가리키지 않게 비움
        if (xioctl(fd_, VIDIOC_QBUF, &held) < 0) {
            std::cerr << "[ERROR] VIDIOC_QBUF(지난 프레임 반납) 실패: " << errnoText() << '\n';
            return false;
        }
    }

    v4l2_buffer buffer{};
    // 프레임이 올 때까지 기다림. 30fps 면 33ms 안에 오므로 2초를 넘기면 카메라 이상으로 봄
    // 첫 프레임 뒤 약 1.3초 멈추는 구간(9/15 확인)이 있어 첫 두 장까지는 5초까지 기다림
    const int timeoutMs = lastSequence_ < 1 ? 5000 : 2000;
    for (;;) {
        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLIN;
        const int ready = poll(&descriptor, 1, timeoutMs);
        if (ready == 0) {
            std::cerr << "[ERROR] 카메라 프레임 대기 " << timeoutMs / 1000 << "초 초과: " << devicePath_ << '\n';
            return false;
        }
        if (ready < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[ERROR] 카메라 poll 실패: " << errnoText() << '\n';
            return false;
        }

        buffer = v4l2_buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &buffer) == 0) break;
        if (errno == EAGAIN) continue;   // poll 은 깨웠는데 아직 준비 전. 다시 기다림
        std::cerr << "[ERROR] VIDIOC_DQBUF 실패: " << errnoText() << '\n';
        return false;
    }

    // 메타데이터. sequence 가 건너뛰면 그 사이 프레임은 카메라·드라이버 쪽에서 빠진 것
    frame.frameSeq = static_cast<std::int64_t>(buffer.sequence);
    frame.droppedBySource = (lastSequence_ >= 0 && frame.frameSeq > lastSequence_ + 1) ? frame.frameSeq - lastSequence_ - 1 : 0;
    frame.droppedByApp = 0;
    lastSequence_ = frame.frameSeq;

    frame.captureTimestampNs = static_cast<std::int64_t>(buffer.timestamp.tv_sec) * 1000000000LL
                             + static_cast<std::int64_t>(buffer.timestamp.tv_usec) * 1000LL;
    const bool monotonic = (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    frame.captureTimestampClock = monotonic ? CaptureTimestampClock::Monotonic : CaptureTimestampClock::Realtime;
    frame.captureTimestampSource = CaptureTimestampSource::V4l2Monotonic;

    // YUYV(픽셀당 2바이트) → BGR. 드라이버 버퍼를 직접 감싸고, 결과는 frame.image 에 새로 씀
    const cv::Mat yuyv(height_, width_, CV_8UC2, buffers_[buffer.index].start, bytesPerLine_);
    cv::cvtColor(yuyv, frame.image, cv::COLOR_YUV2BGR_YUYV);

    // --camera-zero-copy: 버퍼를 다음 read() 까지 들고 있음. GPU 가 이 MMAP 버퍼의 YUYV 를 직접 읽음
    if (holdBuffer_) {
        const MappedBuffer& mapped = buffers_[buffer.index];
        frame.yuyv.data = static_cast<const unsigned char*>(mapped.start);
        frame.yuyv.bufferBytes = mapped.length;
        frame.yuyv.pitch = bytesPerLine_;
        frame.yuyv.width = width_;
        frame.yuyv.height = height_;
        heldIndex_ = static_cast<int>(buffer.index);
        return true;
    }

    // 변환이 끝났으니 버퍼를 드라이버에 돌려줌
    frame.yuyv = YuyvBuffer();
    if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
        std::cerr << "[ERROR] VIDIOC_QBUF(반납) 실패: " << errnoText() << '\n';
        return false;
    }
    return true;
}

int V4l2CameraSource::width() const { return width_; }

int V4l2CameraSource::height() const { return height_; }

double V4l2CameraSource::fps() const { return fps_; }