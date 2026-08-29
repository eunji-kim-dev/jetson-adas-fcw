/*
 * Blackbox ADAS Project - Step 7 Final
 *
 * 주요 기능:
 * 1. YOLOv8n ONNX 객체 검출
 * 2. 차량 그룹 단위 NMS
 * 3. Kalman Filter 기반 다중 객체 추적
 * 4. 내 차선 선행 차량 선택
 * 5. 바운딩박스 높이 변화 기반 TTC Proxy 계산
 * 6. 햇빛/자동 노출 변화와 실제 장면 전환 분리
 * 7. 원거리 도로 crop 보조 추론
 * 8. 포함형/비정상 종횡비 중복 박스 제거
 * 9. 한국어 상태 정보와 경고 문구 출력
 *
 * 주의:
 * TTC-P는 실제 거리 센서로 계산한 TTC가 아니라
 * 영상 속 객체 크기 변화로 추정한 상대적인 TTC Proxy
 */
#include "perception/Classes.hpp"
#include "perception/Detection.hpp"
#include "perception/Frame.hpp"
#include "perception/FrameSource.hpp"
#include "perception/InferenceBackend.hpp"
#include "perception/MultiObjectTracker.hpp"
#include "perception/VideoFileSource.hpp"
#include "perception/YoloDetector.hpp"
#include "adas/LeadSelector.hpp"
#include "adas/WarningPolicy.hpp"
#include "adas/RiskAnalyzer.hpp"
#include "logging/RunLogger.hpp"
#include "RunOptions.hpp"
#include "JetsonEnv.hpp"
#include <opencv2/core.hpp>
#ifdef ADAS_HAS_FREETYPE
#include <opencv2/freetype.hpp>
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * 편집된 테스트 영상의 장면 전환을 감지
 *
 * 이 프로젝트의 입력 영상은 하나의 연속 주행 영상이 아니라
 * 서로 다른 도로 장면이 이어 붙여진 영상
 *
 * 장면이 갑자기 바뀌었는데도 이전 장면의 Tracking ID,
 * TTC-P 이력, 경고 유지 카운터를 그대로 사용하면
 * 새 장면에 아무 차량이 없어도 이전 경고가 잠깐 남을 수 있음
 *
 * 감지 방법:
 * 1. 프레임을 320×180 회색 영상으로 축소
 * 2. 이전 프레임과 평균 절대 차이(MAD) 계산
 * 3. 밝기 히스토그램 상관계수 계산
 * 4. 전역 평균 밝기 이동량(mean shift) 계산
 * 5. 전체 변화량 중 전역 밝기 이동이 차지하는 비율(shiftRatio) 계산
 * 6. 전역 밝기 이동을 제거한 compensatedDiff 계산
 * 7. 구조 변화가 남아 있을 때만 실제 장면 전환으로 판단
 *
 * 동영상 13초 장면 전환 오검출 오류 수정
 * 태양이 구조물 뒤로 가려졌다 다시 나타날 때 자동 노출이 크게 변하면서
 * rawDiff와 히스토그램 변화가 실제 컷처럼 커지는 문제가 있었음
 * 노출 변화는 diff 대부분이 화면 전체의 동일한 밝기 이동에서 발생하므로
 * |meanShift| / rawDiff 비율을 추가해 실제 컷과 구분함
 *   shiftRatio >= 0.50 : 전역 밝기 변화가 지배적 → 노출 변화로 판단
 *   shiftRatio <  0.50 : 구조 변화 비중이 큼 → 장면 전환 후보 유지
 */
class SceneChangeDetector {
public:
    bool update(const cv::Mat& frame, float* meanAbsoluteDifference = nullptr, float* histogramCorrelation = nullptr, float* brightnessShiftRatio = nullptr, float* brightnessCompensatedDifference = nullptr) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(320, 180));

        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

        cv::Mat histogram;
        const int channels[] = {0};
        const int histogramSize[] = {64};
        const float range[] = {0.0F, 256.0F};
        const float* ranges[] = {range};

        cv::calcHist(&gray, 1, channels, cv::Mat(), histogram, 1, histogramSize, ranges, true, false);
        cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);

        // 첫 프레임에는 비교 대상이 없으므로 현재 프레임만 저장하고 장면 전환은 아니라고 반환
        if (previousGray_.empty() || previousHistogram_.empty()) {
            previousGray_ = gray.clone();
            previousHistogram_ = histogram.clone();

            if (meanAbsoluteDifference != nullptr) *meanAbsoluteDifference = 0.0F;
            if (histogramCorrelation != nullptr) *histogramCorrelation = 1.0F;
            if (brightnessShiftRatio != nullptr) *brightnessShiftRatio = 0.0F;
            if (brightnessCompensatedDifference != nullptr) *brightnessCompensatedDifference = 0.0F;

            return false;
        }

        cv::Mat difference;
        cv::absdiff(gray, previousGray_, difference);

        const float frameDifference = static_cast<float>(cv::mean(difference)[0]);
        const float correlation = static_cast<float>(cv::compareHist(previousHistogram_, histogram, cv::HISTCMP_CORREL));

        // 동영상 13초 햇빛/자동 노출 오검출 수정
        // 현재 프레임과 이전 프레임의 평균 밝기 차이를 계산
        const float currentMean = static_cast<float>(cv::mean(gray)[0]);
        const float previousMean = static_cast<float>(cv::mean(previousGray_)[0]);
        const float meanShift = currentMean - previousMean;

        // 전체 프레임 변화량 중
        // 화면 전체가 균일하게 밝아지거나 어두워진 성분의 비율
        const float shiftRatio = std::abs(meanShift) / std::max(frameDifference, 1e-3F);

        // 전역 밝기 이동(meanShift)을 제거한 뒤에도
        // 실제 구조 변화가 얼마나 남아 있는지 계산
        cv::Mat currentFloat, previousFloat;
        gray.convertTo(currentFloat, CV_32F);
        previousGray_.convertTo(previousFloat, CV_32F);

        cv::Mat compensatedDifferenceImage = currentFloat - previousFloat;
        compensatedDifferenceImage -= meanShift;

        cv::Mat compensatedAbsolute;
        cv::absdiff(compensatedDifferenceImage, cv::Scalar::all(0), compensatedAbsolute);

        const float compensatedDifference = static_cast<float>(cv::mean(compensatedAbsolute)[0]);

        if (meanAbsoluteDifference != nullptr) *meanAbsoluteDifference = frameDifference;
        if (histogramCorrelation != nullptr) *histogramCorrelation = correlation;
        if (brightnessShiftRatio != nullptr) *brightnessShiftRatio = shiftRatio;
        if (brightnessCompensatedDifference != nullptr) *brightnessCompensatedDifference = compensatedDifference;

        /*
         * 동영상 전체 프레임 재측정 결과:
         *   실제 컷의 shiftRatio            : 약 0.11 ~ 0.26
         *   햇빛/자동 노출 변화의 shiftRatio : 약 0.59 ~ 0.93
         *
         * 따라서 0.50을 기준으로 전역 밝기 변화와 실제 장면 전환을 분리
         * compensatedDifference는 평균 밝기 이동을 제거한 뒤에도
         * 실제 구조 변화가 남는지 확인하는 보조 조건
         */
        constexpr float diffThreshold = 20.0F;
        constexpr float histogramThreshold = 0.78F;
        constexpr float maximumExposureShiftRatio = 0.50F;
        constexpr float compensatedDiffThreshold = diffThreshold * 0.80F;

        const bool sceneChanged = frameDifference >= diffThreshold &&
                                  correlation <= histogramThreshold &&
                                  shiftRatio < maximumExposureShiftRatio &&
                                  compensatedDifference >= compensatedDiffThreshold;

        previousGray_ = gray.clone();
        previousHistogram_ = histogram.clone();

        return sceneChanged;
    }

private:
    cv::Mat previousGray_;
    cv::Mat previousHistogram_;
};

/*
 * OpenCV 기본 Hershey 글꼴은 한글을 지원하지 않음
 * 따라서 opencv_contrib의 FreeType 모듈과 시스템에 설치된
 * 나눔고딕 또는 Noto Sans CJK 글꼴을 사용
 */
class KoreanTextRenderer {
public:
    bool initialize() {
#ifndef ADAS_HAS_FREETYPE
        // freetype 없는 빌드에서는 OpenCV 기본 Hershey 글꼴로 대체
        fontPath_ = "OpenCV Hershey fallback";
        return true;
#else
        std::vector<std::string> fontCandidates;
        if (const char* customFont = std::getenv("ADAS_KOREAN_FONT")) {
            if (*customFont != '\0') fontCandidates.emplace_back(customFont);
        }
        fontCandidates.insert(fontCandidates.end(), {
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        });

        for (const std::string& fontPath : fontCandidates) {
            if (!std::filesystem::exists(fontPath)) continue;
            try {
                renderer_ = cv::freetype::createFreeType2();
                renderer_->loadFontData(fontPath, 0);
                fontPath_ = fontPath;
                return true;
            } catch (const cv::Exception&) {
                renderer_.release();
            }
        }
        return false;
#endif
    }

    void putText(cv::Mat& image, const std::string& text, const cv::Point& origin,
                 int fontHeight, const cv::Scalar& color, int thickness = -1) const {
#ifdef ADAS_HAS_FREETYPE
        renderer_->putText(image, text, origin, fontHeight, color,
                           thickness, cv::LINE_AA, false);
#else
        cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, fontHeight / 24.0,
                    color, std::max(thickness, 1), cv::LINE_AA);
#endif
    }

    cv::Size getTextSize(const std::string& text, int fontHeight,
                         int thickness, int* baseline) const {
#ifdef ADAS_HAS_FREETYPE
        return renderer_->getTextSize(text, fontHeight, thickness, baseline);
#else
        return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontHeight / 24.0,
                               std::max(thickness, 1), baseline);
#endif
    }

    const std::string& fontPath() const { return fontPath_; }

private:
#ifdef ADAS_HAS_FREETYPE
    cv::Ptr<cv::freetype::FreeType2> renderer_;
#endif
    std::string fontPath_;
};

std::string getRiskNameKorean(RiskLevel level) {
    switch (level) {
        case RiskLevel::Safe: return "안전";
        case RiskLevel::Caution: return "주의";
        case RiskLevel::Danger: return "위험";
    }
    return "알 수 없음";
}

void drawCenteredKoreanText(KoreanTextRenderer& renderer, cv::Mat& frame, const std::string& text, int y, int fontHeight, const cv::Scalar& color) {
    int baseline = 0;
    const cv::Size textSize = renderer.getTextSize(text, fontHeight, -1, &baseline);
    const int x = std::max(0, (frame.cols - textSize.width) / 2);
    renderer.putText(frame, text, cv::Point(x, y), fontHeight, color, -1);
}

int main(int argc, char* argv[]) {
    RunOptions options;
    if (!parseRunOptions(argc, argv, "adas", options)) return 1;
    const std::string& inputPath = options.inputPath;
    const std::string& backendName = options.backendName;

    const std::string modelPath = "models/yolov8n.onnx";
    const std::string inputStem = std::filesystem::path(inputPath).stem().string();
    const std::string outputPath = "results/" + inputStem + "_output.avi";
    const std::string csvPath = "results/" + inputStem + "_frames.csv";

    std::filesystem::create_directories("results");

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[ERROR] YOLO 모델 없음: " << modelPath << '\n';
        return 1;
    }

    KoreanTextRenderer koreanText;
    if (!koreanText.initialize()) {
        std::cerr << "[ERROR] 한글 글꼴 탐색/로드 실패\n필요한 패키지 설치 명령:\nsudo apt install -y libopencv-contrib-dev fonts-nanum\n";
        return 1;
    }
    std::cout << "[SUCCESS] 한글 글꼴: " << koreanText.fontPath() << '\n';

    constexpr float detectorThreshold = 0.10F;
    constexpr float nmsThreshold = 0.45F;

    // 추론 엔진은 --backend로 선택하고, YoloDetector는 backend 종류를 알지 못함
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

    std::ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "[ERROR] CSV 생성 실패: " << csvPath << '\n';
        return 1;
    }

    csvFile << "frame,sceneChanged,numDetections,detections,numTracks,tracks,activeLeadId,riskLevel,ttc\n";

    // 넓은 도로 관심 영역
    // 실제 위험 판단은 아래 egoLaneRoi의 선행 차량 한 대에만 적용
    const std::vector<cv::Point> roadRoi = {
        cv::Point(static_cast<int>(width * 0.37F), static_cast<int>(height * 0.64F)),
        cv::Point(static_cast<int>(width * 0.64F), static_cast<int>(height * 0.64F)),
        cv::Point(static_cast<int>(width * 0.96F), static_cast<int>(height * 0.96F)),
        cv::Point(static_cast<int>(width * 0.04F), static_cast<int>(height * 0.96F))
    };

    // 현재 테스트 영상에서 내 차량이 주행하는 차선 ROI
    const std::vector<cv::Point> egoLaneRoi = {
        cv::Point(static_cast<int>(width * 0.40F), static_cast<int>(height * 0.65F)),
        cv::Point(static_cast<int>(width * 0.48F), static_cast<int>(height * 0.65F)),
        cv::Point(static_cast<int>(width * 0.68F), static_cast<int>(height * 0.95F)),
        cv::Point(static_cast<int>(width * 0.10F), static_cast<int>(height * 0.95F))
    };

    MultiObjectTracker tracker(0.25F, 0.10F, 3, 20);
    LeadSelector leadSelector(roadRoi, egoLaneRoi, sourceFps);
    WarningPolicy warningPolicy(sourceFps, height);
    RiskAnalyzer riskAnalyzer(sourceFps, height, 15, 60);

    /*
     * 장면 전환 직후에는 이전 장면의 추적과 TTC-P 이력이
     * 새 장면으로 이어지지 않도록 약 0.5초 동안 위험 분석을 쉼
     *
     * 이 시간 동안 RiskAnalyzer에는 모든 객체를
     * isLeadTarget=false로 전달해 기존 ID의 기록도 초기화
     */
    const int sceneWarmupFrames = std::max(1, static_cast<int>(std::round(sourceFps * 0.5)));
    int sceneWarmupRemaining = 0;

    // 실행 로그 (results/runs/<run_id>/raw_frame_log.csv, run_summary.json)
    RunMetadata runMetadata;
    runMetadata.runId = options.runId.empty() ? RunLogger::defaultRunId(inputStem, backendName) : options.runId;
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

    SceneChangeDetector sceneChangeDetector;
    Frame captured;
    int processedFrames = 0;
    double totalInferenceMilliseconds = 0.0, totalFullYoloMilliseconds = 0.0, totalFarYoloMilliseconds = 0.0, totalPostprocessMilliseconds = 0.0;

    const auto totalStart = std::chrono::steady_clock::now();

    auto readStart = std::chrono::steady_clock::now();
    while (source.read(captured)) {
        // 프레임이 앱 손에 들어온 순간 (파일: 디코더 반환, 카메라: DQBUF 반환)
        const auto dequeueTime = std::chrono::steady_clock::now();
        cv::Mat& frame = captured.image;
        ++processedFrames;

        // 반드시 ROI, 박스, 글씨를 그리기 전의 원본 프레임으로 장면 전환을 감지
        float sceneDifference = 0.0F, sceneHistogramCorrelation = 1.0F, sceneShiftRatio = 0.0F, sceneCompensatedDifference = 0.0F;
        const bool sceneChanged = sceneChangeDetector.update(frame, &sceneDifference, &sceneHistogramCorrelation, &sceneShiftRatio, &sceneCompensatedDifference);

        if (sceneChanged) {
            // 이전 장면에서 유지 중이던 경고를 즉시 제거
            warningPolicy.reset();
            sceneWarmupRemaining = sceneWarmupFrames;

            // 새 장면에서는 이전 장면의 LEAD/lane 체류/횡이동 이력을 사용하지 않음
            leadSelector.reset();
            // 이전 장면의 Kalman Track과 TTC-P 이력을 새 장면으로 넘기지 않음
            // nextTrackId_는 tracker.reset() 안에서 유지되어 ID는 실행 전체에서 유일함
            tracker.reset(); riskAnalyzer.reset();

            std::cout << "[SCENE CHANGE] frame=" << processedFrames << " diff=" << std::fixed << std::setprecision(2) << sceneDifference << " histogram=" << sceneHistogramCorrelation << " shiftRatio=" << sceneShiftRatio << " compensatedDiff=" << sceneCompensatedDifference << '\n';
        } else if (sceneDifference >= 20.0F && sceneShiftRatio >= 0.50F) {
            // 동영상 13초/17초 햇빛 변화 트러블슈팅 확인용 로그
            // rawDiff는 크지만 shiftRatio가 0.50 이상이면
            // 실제 컷이 아니라 전역 밝기/자동 노출 변화로 걸러진 상태
            std::cout << "[EXPOSURE CHANGE] frame=" << processedFrames << " diff=" << std::fixed << std::setprecision(2) << sceneDifference << " histogram=" << sceneHistogramCorrelation << " shiftRatio=" << sceneShiftRatio << " compensatedDiff=" << sceneCompensatedDifference << '\n';
        }
        const auto sceneEnd = std::chrono::steady_clock::now();

        const bool riskAnalysisEnabled = sceneWarmupRemaining <= 0;
        const auto inferenceStart = std::chrono::steady_clock::now();
        std::vector<Detection> detections;
        DetectionTiming detectionTiming;

        try {
            detections = detector.detect(frame, &detectionTiming);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] 객체 검출 실패: " << error.what() << '\n';
            return 1;
        }

        const double fullYoloMilliseconds = detectionTiming.fullYoloMilliseconds;
        const double farYoloMilliseconds = detectionTiming.farYoloMilliseconds;
        const double postprocessMilliseconds = detectionTiming.postprocessMilliseconds;

        const auto inferenceEnd = std::chrono::steady_clock::now();
        const double inferenceMilliseconds = std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();
        totalInferenceMilliseconds += inferenceMilliseconds;
        totalFullYoloMilliseconds += fullYoloMilliseconds;
        totalFarYoloMilliseconds += farYoloMilliseconds;
        totalPostprocessMilliseconds += postprocessMilliseconds;

        const double averageInferenceMilliseconds = totalInferenceMilliseconds / static_cast<double>(processedFrames);
        std::cout << std::fixed << std::setprecision(2) << "[PERF] frame=" << processedFrames << " | Full YOLO=" << fullYoloMilliseconds << " ms" << " | Far YOLO=" << farYoloMilliseconds << " ms" << " | Post=" << postprocessMilliseconds << " ms" << " | Total=" << inferenceMilliseconds << " ms" << " | AVG=" << averageInferenceMilliseconds << " ms\n";

        const auto trackingStart = std::chrono::steady_clock::now();
        const std::vector<TrackedObject> trackedObjects = tracker.update(detections);
        const auto trackingEnd = std::chrono::steady_clock::now();

        leadSelector.update(trackedObjects, riskAnalysisEnabled);
        const int activeLeadId = leadSelector.activeLeadId();
        const std::unordered_map<int, ObjectGeometry>& geometryById = leadSelector.geometryById();

        cv::Mat overlay = frame.clone();
        cv::fillConvexPoly(overlay, roadRoi, cv::Scalar(0, 255, 0));
        cv::addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame);
        cv::polylines(frame, roadRoi, true, cv::Scalar(0, 180, 0), 2);
        cv::polylines(frame, egoLaneRoi, true, cv::Scalar(255, 255, 255), 3);

        int objectsOnRoad = 0, objectsInEgoLane = 0;
        RiskResult leadRisk;
        bool leadRiskFound = false;

        for (const TrackedObject& trackedObject : trackedObjects) {
            const auto geometryIterator = geometryById.find(trackedObject.trackId);
            if (geometryIterator == geometryById.end()) continue;
            const ObjectGeometry& geometry = geometryIterator->second;

            if (geometry.insideRoad) ++objectsOnRoad;
            if (geometry.insideEgoLane) ++objectsInEgoLane;

            // ego lane 차량은 LEAD가 되기 전부터 TTC-P 샘플을 축적한다.
            // 실제 CAUTION/DANGER 단계 판정은 activeLeadId 한 대에만 수행한다.
            const bool isAnalysisTarget = riskAnalysisEnabled && geometry.laneHeld && isVehicleClass(trackedObject.classId);
            const bool isLeadTarget = riskAnalysisEnabled && trackedObject.trackId == activeLeadId && geometry.laneHeld;
            
            const RiskResult rawRisk = riskAnalyzer.update(trackedObject, isAnalysisTarget, isLeadTarget, processedFrames);
            
            const RiskResult risk = warningPolicy.applyGeometryGate(rawRisk, geometry, isLeadTarget);

            if (isLeadTarget) {
                leadRisk = risk;
                leadRiskFound = true;
            }

            cv::Scalar boxColor;
            if (!geometry.insideRoad) boxColor = cv::Scalar(0, 165, 255);
            else if (!geometry.insideEgoLane) boxColor = cv::Scalar(255, 255, 0);
            else if (!isLeadTarget) boxColor = cv::Scalar(255, 180, 0);
            else if (risk.level == RiskLevel::Danger) boxColor = cv::Scalar(0, 0, 255);
            else if (risk.level == RiskLevel::Caution) boxColor = cv::Scalar(0, 255, 255);
            else boxColor = cv::Scalar(0, 255, 0);

            cv::rectangle(frame, trackedObject.box, boxColor, 3);
            cv::circle(frame, geometry.groundPoint, 6, boxColor, -1);

            std::string label = getClassName(trackedObject.classId) + " ID:" + std::to_string(trackedObject.trackId) + " " + cv::format("%.2f", trackedObject.confidence);

            if (isLeadTarget) {
                label = "LEAD " + label + " " + RiskAnalyzer::toString(risk.level);
                if (risk.valid && std::isfinite(risk.ttcSeconds)) label += cv::format(" TTC-P:%.1fs", risk.ttcSeconds);
                else label += " TTC-P:--";
            } else if (geometry.insideEgoLane) {
                label += " EGO-LANE";
            } else if (geometry.insideRoad) {
                label += " SIDE";
            }

            int baseline = 0;
            const cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
            const int labelX = std::clamp(trackedObject.box.x, 0, std::max(width - labelSize.width - 10, 0));
            const int labelTop = std::max(trackedObject.box.y, labelSize.height + 10);

            cv::rectangle(frame, cv::Point(labelX, labelTop - labelSize.height - 10), cv::Point(labelX + labelSize.width + 10, labelTop), boxColor, cv::FILLED);
            cv::putText(frame, label, cv::Point(labelX + 5, labelTop - 5), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }

        /*
         * LEAD 차량 시각화는 항상 마지막에 한 번 더 그려 최상위 레이어로 유지
         *
         * trackedObjects 순서에 따라 LEAD가 먼저 그려지면 뒤에 그려지는 다른
         * 차량의 bbox/라벨이 LEAD 박스를 가릴 수 있음
         * 판정 로직은 그대로 두고, 모든 객체 렌더링이 끝난 뒤
         * 현재 LEAD의 bbox/라벨만 다시 그림
         */
        if (riskAnalysisEnabled && activeLeadId >= 0 && leadRiskFound) {
            const auto leadIterator = std::find_if(trackedObjects.begin(), trackedObjects.end(), [&](const TrackedObject& candidate) { return candidate.trackId == activeLeadId; });
            const auto leadGeometryIterator = leadIterator != trackedObjects.end() ? geometryById.find(activeLeadId) : geometryById.end();

            if (leadIterator != trackedObjects.end() && leadGeometryIterator != geometryById.end() && leadGeometryIterator->second.insideEgoLane) {
                const TrackedObject& trackedObject = *leadIterator;
                const ObjectGeometry& geometry = leadGeometryIterator->second;
                cv::Scalar leadBoxColor;

                if (leadRisk.level == RiskLevel::Danger) leadBoxColor = cv::Scalar(0, 0, 255);
                else if (leadRisk.level == RiskLevel::Caution) leadBoxColor = cv::Scalar(0, 255, 255);
                else leadBoxColor = cv::Scalar(0, 255, 0);

                cv::rectangle(frame, trackedObject.box, leadBoxColor, 3);
                cv::circle(frame, geometry.groundPoint, 6, leadBoxColor, -1);

                std::string leadLabel = "LEAD " + getClassName(trackedObject.classId) + " ID:" + std::to_string(trackedObject.trackId) + " " + cv::format("%.2f", trackedObject.confidence) + " " + RiskAnalyzer::toString(leadRisk.level);
                if (leadRisk.valid && std::isfinite(leadRisk.ttcSeconds)) leadLabel += cv::format(" TTC-P:%.1fs", leadRisk.ttcSeconds);
                else leadLabel += " TTC-P:--";

                int baseline = 0;
                const cv::Size labelSize = cv::getTextSize(leadLabel, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
                const int labelX = std::clamp(trackedObject.box.x, 0, std::max(width - labelSize.width - 10, 0));
                const int labelTop = std::max(trackedObject.box.y, labelSize.height + 10);

                cv::rectangle(frame, cv::Point(labelX, labelTop - labelSize.height - 10), cv::Point(labelX + labelSize.width + 10, labelTop), leadBoxColor, cv::FILLED);
                cv::putText(frame, leadLabel, cv::Point(labelX + 5, labelTop - 5), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
            }
        }

        riskAnalyzer.removeStaleTracks(processedFrames);

        warningPolicy.update(riskAnalysisEnabled, sceneChanged, leadRiskFound, activeLeadId, leadRisk.level);
        // 이 프레임의 판정(경고 배너)이 확정된 순간
        const auto decisionTime = std::chrono::steady_clock::now();

        // 장면 전환 직후의 위험 분석 대기 시간을 한 프레임 줄임
        if (sceneWarmupRemaining > 0) --sceneWarmupRemaining;

        koreanText.putText(frame, "프레임: " + std::to_string(processedFrames), cv::Point(30, 40), 25, cv::Scalar(255, 255, 255));
        koreanText.putText(frame, "추론 시간: " + std::string(cv::format("%.1f ms", inferenceMilliseconds)), cv::Point(30, 72), 25, cv::Scalar(255, 255, 255));
        koreanText.putText(frame, "도로 객체: " + std::to_string(objectsOnRoad) + "  내 차선: " + std::to_string(objectsInEgoLane), cv::Point(30, 104), 25, cv::Scalar(255, 255, 255));

        std::string leadStatus = "선행 차량: 없음";
        if (activeLeadId >= 0 && leadRiskFound) {
            leadStatus = "선행 차량 ID:" + std::to_string(activeLeadId) + "  상태: " + getRiskNameKorean(leadRisk.level);
            if (leadRisk.valid && std::isfinite(leadRisk.ttcSeconds)) leadStatus += "  TTC-P:" + std::string(cv::format("%.1f초", leadRisk.ttcSeconds));
            else leadStatus += "  TTC-P:--";
        }

        koreanText.putText(frame, leadStatus, cv::Point(30, 136), 25, cv::Scalar(255, 255, 255));

        if (warningPolicy.bannerLevel() == RiskLevel::Danger) drawCenteredKoreanText(koreanText, frame, "위험: 충돌 가능성 높음", 58, 34, cv::Scalar(0, 0, 255));
        else if (warningPolicy.bannerLevel() == RiskLevel::Caution) drawCenteredKoreanText(koreanText, frame, "주의: 선행 차량 접근 중", 58, 31, cv::Scalar(0, 255, 255));

        
        auto sortedDetections = detections;
        auto sortedTracks = trackedObjects;

        std::sort(sortedDetections.begin(), sortedDetections.end(),
          [](const Detection& first, const Detection& second) {
              return std::tie(first.classId, first.box.x, first.box.y, first.box.width, first.box.height)
                   < std::tie(second.classId, second.box.x, second.box.y, second.box.width, second.box.height);
          });

        std::sort(sortedTracks.begin(), sortedTracks.end(),
          [](const TrackedObject& first, const TrackedObject& second) {
              return first.trackId < second.trackId;
          });

        std::ostringstream detectionStream;

        detectionStream << std::fixed << std::setprecision(4);

        for (std::size_t i = 0; i < sortedDetections.size(); ++i) {
            const Detection& detection = sortedDetections[i];
            if (i > 0) detectionStream << ';';

            detectionStream << detection.classId << ':'
                            << detection.box.x << ':'
                            << detection.box.y << ':'
                            << detection.box.width << ':'
                            << detection.box.height << ':'
                            << detection.confidence;
        }

        std::ostringstream trackStream;

        for (std::size_t i = 0; i < sortedTracks.size(); ++i) {
            const TrackedObject& track = sortedTracks[i];
            if (i > 0) trackStream << ';';

            trackStream << track.trackId << ':'
                        << track.classId << ':'
                        << track.box.x << ':'
                        << track.box.y << ':'
                        << track.box.width << ':'
                        << track.box.height;
        }

        const std::string riskLevel = leadRiskFound ? RiskAnalyzer::toString(leadRisk.level) : "NONE";

        std::string ttcText = "inf";

        if (leadRiskFound && std::isfinite(leadRisk.ttcSeconds)) {
            std::ostringstream ttcStream;
            ttcStream << std::fixed << std::setprecision(3) << leadRisk.ttcSeconds;
            ttcText = ttcStream.str();
        }

        // CSV용 데이터 준비
        csvFile
            << processedFrames << ','
            << (sceneChanged ? 1 : 0) << ','
            << detections.size() << ','
            << detectionStream.str() << ','
            << trackedObjects.size() << ','
            << trackStream.str() << ','
            << activeLeadId << ','
            << riskLevel << ','
            << ttcText
            << '\n';

        writer.write(frame);
        const auto outputEnd = std::chrono::steady_clock::now();

        // 실행 로그 한 행 — 어느 측정 구간에도 포함되지 않도록 모든 시각을 잰 뒤에 기록
        FrameRecord record;
        record.frame = processedFrames;
        record.frameSeq = captured.frameSeq;
        record.captureTsNs = captured.captureTimestampNs;
        record.captureTsClock = toString(captured.captureTimestampClock);
        record.captureTsSource = toString(captured.captureTimestampSource);
        record.dequeueTsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(dequeueTime.time_since_epoch()).count();
        record.decisionTsNs = std::chrono::duration_cast<std::chrono::nanoseconds>(decisionTime.time_since_epoch()).count();
        record.captureMs = std::chrono::duration<double, std::milli>(dequeueTime - readStart).count();
        record.sceneMs = std::chrono::duration<double, std::milli>(sceneEnd - dequeueTime).count();
        record.preprocessFullMs = detectionTiming.fullInference.preprocessMilliseconds;
        record.inferenceFullMs = detectionTiming.fullInference.inferenceMilliseconds;
        record.postprocessFullMs = detectionTiming.fullInference.postprocessMilliseconds;
        record.preprocessCropMs = detectionTiming.farInference.preprocessMilliseconds;
        record.inferenceCropMs = detectionTiming.farInference.inferenceMilliseconds;
        record.postprocessCropMs = detectionTiming.farInference.postprocessMilliseconds;
        record.mergeMs = postprocessMilliseconds;
        record.detectMs = inferenceMilliseconds;
        record.trackingMs = std::chrono::duration<double, std::milli>(trackingEnd - trackingStart).count();
        record.decisionMs = std::chrono::duration<double, std::milli>(decisionTime - trackingEnd).count();
        record.totalProcessingMs = std::chrono::duration<double, std::milli>(decisionTime - dequeueTime).count();
        record.outputMs = std::chrono::duration<double, std::milli>(outputEnd - decisionTime).count();
        record.deadlineMiss = record.totalProcessingMs > runMetadata.deadlineMs;
        record.detections = static_cast<int>(detections.size());
        record.tracks = static_cast<int>(trackedObjects.size());
        record.leadId = activeLeadId;
        record.leadFound = leadRiskFound;
        if (leadRiskFound && leadRisk.valid && std::isfinite(leadRisk.ttcSeconds)) record.ttcP = leadRisk.ttcSeconds;
        record.riskState = riskLevel;
        record.warningState = RiskAnalyzer::toString(warningPolicy.bannerLevel());
        record.sceneChanged = sceneChanged;
        runLogger.writeFrame(record);

        // --measured-frames: warmup + 측정 프레임 수를 채우면 정지 (반복 측정용)
        if (options.measuredFrames > 0 && processedFrames >= options.warmupFrames + options.measuredFrames) break;

        readStart = std::chrono::steady_clock::now();
    }

    runLogger.setTemperatureEnd(
        jetson_env::readSocTemperatureC()
    );

    runLogger.finish();

    const auto totalEnd = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(totalEnd - totalStart).count();
    const double processingFps = elapsedSeconds > 0.0 ? static_cast<double>(processedFrames) / elapsedSeconds : 0.0;
    const double averageInferenceMilliseconds = processedFrames > 0 ? totalInferenceMilliseconds / static_cast<double>(processedFrames) : 0.0;
    const double averageFullYoloMilliseconds = processedFrames > 0 ? totalFullYoloMilliseconds / static_cast<double>(processedFrames) : 0.0;
    const double averageFarYoloMilliseconds = processedFrames > 0 ? totalFarYoloMilliseconds / static_cast<double>(processedFrames) : 0.0;
    const double averagePostprocessMilliseconds = processedFrames > 0 ? totalPostprocessMilliseconds / static_cast<double>(processedFrames) : 0.0;
    const double inferenceFps = averageInferenceMilliseconds > 0.0 ? 1000.0 / averageInferenceMilliseconds : 0.0;

    writer.release();
    csvFile.close();

    std::cout << '\n';
    std::cout << "[SUCCESS] Step 7 장면 전환/검출 오류 보완 영상 생성 완료\n";
    std::cout << "처리 프레임 수: " << processedFrames << '\n';
    std::cout << "전체 YOLO 평균: " << std::fixed << std::setprecision(2) << averageFullYoloMilliseconds << " ms\n";
    std::cout << "원거리 YOLO 평균: " << std::fixed << std::setprecision(2) << averageFarYoloMilliseconds << " ms\n";
    std::cout << "후처리 평균: " << std::fixed << std::setprecision(3) << averagePostprocessMilliseconds << " ms\n";
    std::cout << "평균 추론 시간: " << std::fixed << std::setprecision(2) << averageInferenceMilliseconds << " ms\n";
    std::cout << "추론 기준 FPS: " << std::fixed << std::setprecision(2) << inferenceFps << " FPS\n";
    std::cout << "전체 처리 속도: " << std::fixed << std::setprecision(2) << processingFps << " FPS\n";
    std::cout << "결과 파일: " << outputPath << '\n';
    std::cout << "실행 로그: " << runLogger.runDirectory() << '\n';

    return 0;
}
