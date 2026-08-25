/*
 * Perception 단독 실행 데모
 *
 * adas_fcw를 전혀 링크하지 않고 perception_core만으로
 * Detection + Tracking 파이프라인이 동작하는 것을 보이는 실행 파일
 *
 * FCW 개념(LEAD, TTC-P, 위험도)이 없으므로 결과 영상에는
 * 클래스/Track ID만 표시됨
 *
 * 주의: SceneChangeDetector가 없으므로 편집된 테스트 영상에서는
 * 장면 전환 시 Track ID가 이어질 수 있음 (의도된 동작)
 */
#include "perception/Classes.hpp"
#include "perception/Detection.hpp"
#include "perception/MultiObjectTracker.hpp"
#include "perception/YoloDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    const std::string inputPath = argc >= 2 ? argv[1] : "videos/input.mp4";
    const std::string modelPath = "models/yolov8n.onnx";
    const std::string inputStem = std::filesystem::path(inputPath).stem().string();
    const std::string outputPath = "results/" + inputStem + "_perception.avi";

    std::filesystem::create_directories("results");

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[ERROR] YOLO 모델 없음: " << modelPath << '\n';
        return 1;
    }

    constexpr float detectorThreshold = 0.10F;
    constexpr float nmsThreshold = 0.45F;

    std::unique_ptr<YoloDetector> detectorPtr;
    try {
        detectorPtr = std::make_unique<YoloDetector>(modelPath, detectorThreshold, nmsThreshold);
    } catch (const cv::Exception& error) {
        std::cerr << "[ERROR] YOLO 모델 로드 실패\n" << error.what() << '\n';
        return 1;
    }
    YoloDetector& detector = *detectorPtr;

    cv::VideoCapture capture(inputPath);
    if (!capture.isOpened()) {
        std::cerr << "[ERROR] 영상 열기 실패: " << inputPath << '\n';
        return 1;
    }

    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double sourceFps = capture.get(cv::CAP_PROP_FPS);
    if (sourceFps <= 0.0) sourceFps = 30.0;

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), sourceFps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] 결과 영상 생성 실패: " << outputPath << '\n';
        return 1;
    }

    MultiObjectTracker tracker(0.25F, 0.10F, 3, 20);
    cv::Mat frame;
    int processedFrames = 0;

    while (capture.read(frame)) {
        ++processedFrames;

        std::vector<Detection> detections;
        try {
            detections = detector.detect(frame);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] 객체 검출 실패: " << error.what() << '\n';
            return 1;
        }

        const std::vector<TrackedObject> trackedObjects = tracker.update(detections);

        for (const TrackedObject& trackedObject : trackedObjects) {
            cv::rectangle(frame, trackedObject.box, cv::Scalar(0, 255, 0), 2);
            const std::string label = getClassName(trackedObject.classId) + " ID:" + std::to_string(trackedObject.trackId);
            cv::putText(frame, label, cv::Point(trackedObject.box.x, std::max(trackedObject.box.y - 5, 12)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        if (processedFrames % 100 == 0) {
            std::cout << "frame=" << processedFrames << " detections=" << detections.size() << " tracks=" << trackedObjects.size() << '\n';
        }

        writer.write(frame);
    }

    capture.release();
    writer.release();

    std::cout << "[SUCCESS] perception 단독 실행 완료: " << processedFrames << " 프레임 처리\n";
    std::cout << "결과 파일: " << outputPath << '\n';
    return 0;
}
