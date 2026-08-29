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
#include "perception/Frame.hpp"
#include "perception/FrameSource.hpp"
#include "perception/InferenceBackend.hpp"
#include "perception/MultiObjectTracker.hpp"
#include "perception/VideoFileSource.hpp"
#include "perception/YoloDetector.hpp"
#include "logging/RunLogger.hpp"
#include "RunOptions.hpp"
#include "JetsonEnv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    RunOptions options;
    if (!parseRunOptions(argc, argv, "perception_demo", options)) return 1;
    const std::string& inputPath = options.inputPath;
    const std::string& backendName = options.backendName;

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
        std::unique_ptr<InferenceBackend> backend = createInferenceBackend(backendName, modelPath, detectorThreshold, nmsThreshold);
        detectorPtr = std::make_unique<YoloDetector>(std::move(backend), nmsThreshold);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] 추론 backend 생성 실패 (backend=" << backendName << ")\n" << error.what() << '\n';
        return 1;
    }
    YoloDetector& detector = *detectorPtr;
    std::cout << "[INFO] 추론 backend: " << backendName << '\n';

    // 영상 입력은 FrameSource 인터페이스 뒤에 둔다
    // 실시간 카메라(CameraSource)로 바꿀 때 이 생성부만 교체하면 됨
    std::unique_ptr<FrameSource> sourcePtr;
    try {
        sourcePtr = std::make_unique<VideoFileSource>(inputPath);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 1;
    }
    FrameSource& source = *sourcePtr;

    const int width = source.width();
    const int height = source.height();
    const double sourceFps = source.fps();

    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), sourceFps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "[ERROR] 결과 영상 생성 실패: " << outputPath << '\n';
        return 1;
    }

    // 실행 로그 — adas 와 같은 스키마, FCW 관련 컬럼은 빈 칸
    RunMetadata runMetadata;
    runMetadata.runId = options.runId.empty() ? RunLogger::defaultRunId(inputStem + "_perception", backendName) : options.runId;
    runMetadata.powerMode = options.powerMode;
    runMetadata.backend = backendName;
    runMetadata.precision = "fp32";
    runMetadata.temperatureStartC = jetson_env::readSocTemperatureC();
    runMetadata.jetsonClocks = jetson_env::readJetsonClocksActive();
    runMetadata.opencvVersion = CV_VERSION;
    runMetadata.inputSource = "video_file";
    runMetadata.inputName = inputPath;
    runMetadata.inputHash = RunLogger::hashFile(inputPath);
    runMetadata.resolution = std::to_string(width) + "x" + std::to_string(height);
    runMetadata.sourceFps = sourceFps;
    runMetadata.captureTsClock = toString(CaptureTimestampClock::Stream);
    runMetadata.captureTsSource = toString(CaptureTimestampSource::VideoPts);
    runMetadata.model = modelPath;
    runMetadata.modelHash = RunLogger::hashFile(modelPath);
    runMetadata.fullCropStrategy = "full+crop_every_frame";
    runMetadata.detectionInterval = 1;
    runMetadata.confidenceThreshold = detectorThreshold;
    runMetadata.nmsThreshold = nmsThreshold;
    runMetadata.warmupFrames = options.warmupFrames;
    runMetadata.measuredFrames = options.measuredFrames;
    runMetadata.deadlineMs = 1000.0 / sourceFps;

    std::unique_ptr<RunLogger> runLoggerPtr;
    try {
        runLoggerPtr = std::make_unique<RunLogger>("results/runs", runMetadata);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] 실행 로그 생성 실패\n" << error.what() << '\n';
        return 1;
    }
    RunLogger& runLogger = *runLoggerPtr;
    std::cout << "[INFO] 실행 로그: " << runLogger.runDirectory() << " (git " << runLogger.gitCommit().substr(0, 12) << (runLogger.gitDirty() ? ", dirty" : "") << ")\n";
    if (runLogger.gitDirty()) std::cout << "[WARN] working tree 가 dirty 상태 — 이 실행의 측정값은 baseline 으로 쓰지 않음\n";

    MultiObjectTracker tracker(0.25F, 0.10F, 3, 20);
    Frame captured;
    int processedFrames = 0;

    auto readStart = std::chrono::steady_clock::now();
    while (source.read(captured)) {
        const auto dequeueTime = std::chrono::steady_clock::now();
        cv::Mat& frame = captured.image;
        ++processedFrames;

        std::vector<Detection> detections;
        DetectionTiming detectionTiming;
        const auto detectStart = std::chrono::steady_clock::now();
        try {
            detections = detector.detect(frame, &detectionTiming);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] 객체 검출 실패: " << error.what() << '\n';
            return 1;
        }
        const auto detectEnd = std::chrono::steady_clock::now();

        const std::vector<TrackedObject> trackedObjects = tracker.update(detections);
        // perception 단독 실행에서는 추적 결과가 곧 이 프레임의 출력
        const auto decisionTime = std::chrono::steady_clock::now();

        for (const TrackedObject& trackedObject : trackedObjects) {
            cv::rectangle(frame, trackedObject.box, cv::Scalar(0, 255, 0), 2);
            const std::string label = getClassName(trackedObject.classId) + " ID:" + std::to_string(trackedObject.trackId);
            cv::putText(frame, label, cv::Point(trackedObject.box.x, std::max(trackedObject.box.y - 5, 12)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        if (processedFrames % 100 == 0) {
            std::cout << "frame=" << processedFrames << " detections=" << detections.size() << " tracks=" << trackedObjects.size() << '\n';
        }

        writer.write(frame);
        const auto outputEnd = std::chrono::steady_clock::now();

        FrameRecord record;
        record.frame = processedFrames;
        record.frameSeq = captured.frameSeq;
        record.captureTsNs = captured.captureTimestampNs;
        record.captureTsClock = toString(captured.captureTimestampClock);
        record.captureTsSource = toString(captured.captureTimestampSource);
        record.dequeueTsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(dequeueTime.time_since_epoch()).count();
        record.decisionTsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(decisionTime.time_since_epoch()).count();
        record.captureMs = std::chrono::duration<double, std::milli>(dequeueTime - readStart).count();
        record.preprocessFullMs = detectionTiming.fullInference.preprocessMilliseconds;
        record.inferenceFullMs = detectionTiming.fullInference.inferenceMilliseconds;
        record.postprocessFullMs = detectionTiming.fullInference.postprocessMilliseconds;
        record.preprocessCropMs = detectionTiming.farInference.preprocessMilliseconds;
        record.inferenceCropMs = detectionTiming.farInference.inferenceMilliseconds;
        record.postprocessCropMs = detectionTiming.farInference.postprocessMilliseconds;
        record.mergeMs = detectionTiming.postprocessMilliseconds;
        record.detectMs = std::chrono::duration<double, std::milli>(detectEnd - detectStart).count();
        record.trackingMs = std::chrono::duration<double, std::milli>(decisionTime - detectEnd).count();
        record.totalProcessingMs = std::chrono::duration<double, std::milli>(decisionTime - dequeueTime).count();
        record.outputMs = std::chrono::duration<double, std::milli>(outputEnd - decisionTime).count();
        record.deadlineMiss = record.totalProcessingMs > runMetadata.deadlineMs;
        record.detections = static_cast<int>(detections.size());
        record.tracks = static_cast<int>(trackedObjects.size());
        runLogger.writeFrame(record);

        if (options.measuredFrames > 0 && processedFrames >= options.warmupFrames + options.measuredFrames) break;

        readStart = std::chrono::steady_clock::now();
    }

    runLogger.setTemperatureEnd(
        jetson_env::readSocTemperatureC()
    );

    runLogger.finish();
    writer.release();

    std::cout << "[SUCCESS] perception 단독 실행 완료: " << processedFrames << " 프레임 처리\n";
    std::cout << "결과 파일: " << outputPath << '\n';
    std::cout << "실행 로그: " << runLogger.runDirectory() << '\n';
    return 0;
}
