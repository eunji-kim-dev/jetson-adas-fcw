#include "perception/Detection.hpp"
#include "perception/MultiObjectTracker.hpp"

#include <opencv2/videoio.hpp>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    const std::string inputPath = argc >= 2 ? argv[1] : "videos/input.mp4";

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] 영상 열기 실패: " << inputPath << '\n';
        return 1;
    }

    MultiObjectTracker tracker(0.25F, 0.10F, 3, 20);
    cv::Mat frame;
    int processedFrames = 0;

    while (capture.read(frame)) {
        ++processedFrames;

        // TODO: STEP 4에서 perception::YoloDetector로 교체
        const std::vector<Detection> detections;
        const std::vector<TrackedObject> trackedObjects = tracker.update(detections);

        if (processedFrames % 100 == 0) {
            std::cout << "frame=" << processedFrames << " tracks=" << trackedObjects.size() << '\n';
        }
    }

    capture.release();
    std::cout << "[SUCCESS] adas_fcw 미링크 상태로 " << processedFrames << " 프레임 처리\n";
    return 0;
}