#include "perception/ThreadedFrameSource.hpp"

#include <utility>

ThreadedFrameSource::ThreadedFrameSource(std::unique_ptr<FrameSource> inner)
    : inner_(std::move(inner)), hasLatest_(false), ended_(false), stop_(false), pendingAppDrops_(0), pendingSourceDrops_(0) {
    thread_ = std::thread(&ThreadedFrameSource::captureLoop, this);
}

ThreadedFrameSource::~ThreadedFrameSource() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    // 캡처 스레드가 read() 하나를 마치면 stop_ 을 보고 나옴 (카메라는 최대 2초, 파일은 바로)
    if (thread_.joinable()) thread_.join();
}

void ThreadedFrameSource::captureLoop() {
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) return;
        }

        // 매번 새 Frame 으로 읽음 → 처리 스레드가 들고 있는 image 와 버퍼를 공유하지 않음
        Frame frame;
        const bool ok = inner_->read(frame);

        std::lock_guard<std::mutex> lock(mutex_);
        if (!ok) {
            ended_ = true;
            frameReady_.notify_all();
            return;
        }
        if (hasLatest_) {
            // 처리 쪽이 아직 안 가져간 프레임을 새 것으로 덮어씀 = 프로그램 쪽 버림
            ++pendingAppDrops_;
            pendingSourceDrops_ += latest_.droppedBySource;
        }
        latest_ = std::move(frame);
        hasLatest_ = true;
        frameReady_.notify_all();
    }
}

bool ThreadedFrameSource::read(Frame& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    frameReady_.wait(lock, [this] { return hasLatest_ || ended_; });
    if (!hasLatest_) return false;

    frame = std::move(latest_);
    hasLatest_ = false;
    frame.droppedByApp = pendingAppDrops_;
    frame.droppedBySource += pendingSourceDrops_;
    pendingAppDrops_ = 0;
    pendingSourceDrops_ = 0;
    return true;
}

int ThreadedFrameSource::width() const { return inner_->width(); }

int ThreadedFrameSource::height() const { return inner_->height(); }

double ThreadedFrameSource::fps() const { return inner_->fps(); }