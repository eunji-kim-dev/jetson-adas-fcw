#pragma once

#include "perception/FrameSource.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/*
 * V4L2 USB 카메라 입력 (MMAP 스트리밍, YUYV → BGR)
 *
 * - 열 때 포맷(YUYV, width×height)·fps 를 설정하고 자동 노출을 끔
 *   (v4l2-ctl 의 auto_exposure=1 과 같음. 켜 두면 어두울 때 25fps 로 떨어짐 — 9/15 확인)
 *   재부팅하면 기본값(3, 자동)으로 돌아가므로 프로그램이 열 때마다 설정함
 * - frameSeq           : v4l2_buffer.sequence 그대로. 드라이버가 프레임을 놓치면 번호가 건너뜀
 * - captureTimestampNs : v4l2_buffer.timestamp (노출 시작 시각) 를 ns 로 변환
 * - captureTimestampClock  : 버퍼 플래그가 MONOTONIC 이면 Monotonic, 아니면 Realtime
 * - captureTimestampSource : 항상 V4l2Monotonic
 * - droppedBySource    : 직전에 받은 sequence 와의 차이 - 1 (첫 프레임은 0). 카메라·드라이버 쪽 누락
 *
 * 색 변환(cvtColor)은 read() 안에서 함. 버퍼를 드라이버에 돌려주기 전에 변환을 끝내야 하기 때문임
 * 그래서 capture_ms 에는 프레임 대기 + 색 변환 시간이 같이 들어감
 */
class V4l2CameraSource : public FrameSource {
public:
    // 장치를 못 열거나 포맷·fps·자동 노출 설정에 실패하면 std::runtime_error
    V4l2CameraSource(const std::string& devicePath, int width, int height, double fps, unsigned bufferCount = 4);
    ~V4l2CameraSource() override;

    V4l2CameraSource(const V4l2CameraSource&) = delete;
    V4l2CameraSource& operator=(const V4l2CameraSource&) = delete;

    // 다음 프레임을 기다렸다가(최대 2초) BGR 로 바꿔 frame 에 채움. 시간 초과·오류면 false
    bool read(Frame& frame) override;

    // 드라이버가 실제로 적용한 값 (요청과 다를 수 있음)
    int width() const override;
    int height() const override;
    double fps() const override;

    const std::string& devicePath() const { return devicePath_; }

    // GPU 전처리 v2 (--camera-zero-copy). 켜면 read() 가 버퍼를 드라이버에 바로 돌려주지 않고 다음 read() 첫머리에 돌려줌
    // 그동안 frame.yuyv 가 그 MMAP 버퍼를 가리켜 GPU 가 복사 없이 읽음. 들고 있는 1개를 뺀 나머지 버퍼로 촬영은 계속됨
    // 캡처 스레드(ThreadedFrameSource)와 같이 쓰면 안 됨 — 처리 중에 다음 read() 가 불려 버퍼가 돌아감 (RunOptions 가 막음)
    void setHoldBuffer(bool hold) { holdBuffer_ = hold; }

private:
    struct MappedBuffer {
        void* start = nullptr;
        std::size_t length = 0;
    };

    void setFormat(int width, int height);
    void setFrameRate(double fps);
    void disableAutoExposure();
    void mapBuffers(unsigned bufferCount);
    void startStreaming();
    void stopStreaming();   // STREAMOFF + munmap. 여러 번 불러도 됨

    std::string devicePath_;
    int fd_;
    int width_;
    int height_;
    std::size_t bytesPerLine_;
    double fps_;
    std::vector<MappedBuffer> buffers_;
    bool streaming_;
    std::int64_t lastSequence_;   // 아직 한 장도 못 받았으면 -1
    bool holdBuffer_ = false;     // setHoldBuffer
    int heldIndex_ = -1;          // 다음 read() 때 드라이버에 돌려줄 버퍼 번호. 없으면 -1
};