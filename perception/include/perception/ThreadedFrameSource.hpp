#pragma once

#include "perception/FrameSource.hpp"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

/*
 * 캡처 스레드 분리 (최신 프레임 우선)
 *
 * 안쪽 FrameSource 를 별도 스레드에서 계속 읽고, 최신 프레임 한 장만 남김
 * 처리 쪽이 read() 를 부르면 그 시점의 최신 프레임을 가져감
 * 처리가 느려서 가져가기 전에 새 프레임이 오면 이전 것을 버림 → droppedByApp 로 셈
 *
 * - 캡처 스레드는 매번 새 Frame 으로 읽으므로, 처리 중인 image 와 다음 image 는 서로 다른 버퍼임
 * - 안쪽 소스가 준 droppedBySource 는 버린 프레임 몫까지 합쳐서 넘김 (누락 수가 사라지지 않음)
 * - 영상 파일에 씌우면 디코드가 처리보다 빨라 대부분 건너뜀 → 동작 확인용으로만 씀 (golden 비교 불가)
 * - 옵션이 꺼져 있으면 이 클래스를 거치지 않으므로 기존 경로는 그대로임
 */
class ThreadedFrameSource : public FrameSource {
public:
    explicit ThreadedFrameSource(std::unique_ptr<FrameSource> inner);
    ~ThreadedFrameSource() override;   // 캡처 스레드를 멈추고 기다림

    ThreadedFrameSource(const ThreadedFrameSource&) = delete;
    ThreadedFrameSource& operator=(const ThreadedFrameSource&) = delete;

    // 새 프레임이 올 때까지 기다렸다가 가져감. 안쪽 소스가 끝났고 남은 프레임이 없으면 false
    bool read(Frame& frame) override;

    int width() const override;
    int height() const override;
    double fps() const override;

private:
    void captureLoop();

    std::unique_ptr<FrameSource> inner_;
    std::mutex mutex_;
    std::condition_variable frameReady_;
    Frame latest_;                      // 처리 쪽이 아직 안 가져간 최신 프레임
    bool hasLatest_;
    bool ended_;                        // 안쪽 소스가 false 를 돌려줌
    bool stop_;                         // 소멸자가 세움
    std::int64_t pendingAppDrops_;      // 덮어써서 버린 프레임 수 (다음 read 에 실어 보냄)
    std::int64_t pendingSourceDrops_;   // 버린 프레임들이 갖고 있던 droppedBySource 합
    std::thread thread_;                // 다른 멤버가 준비된 뒤 시작하도록 맨 마지막에 둠
};