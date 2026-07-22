/*
 * Blackbox ADAS Project - Step 7 Final
 *
 * 주요 기능:
 * 1. YOLOv8n ONNX 객체 검출
 * 2. 차량 그룹 단위 NMS
 * 3. Kalman Filter 기반 다중 객체 추적
 * 4. 내 차선 선행 차량 선택
 * 5. 바운딩박스 높이 변화 기반 TTC Proxy 계산
 * 6. 우측 탱크로리의 train 오분류를 truck으로 제한 복구
 * 7. 한국어 상태 정보와 경고 문구 출력
 *
 * 주의:
 * TTC-P는 실제 거리 센서로 계산한 TTC가 아니라
 * 영상 속 객체 크기 변화로 추정한 상대적인 TTC Proxy입니다.
 */

#include "Detection.hpp"
#include "MultiObjectTracker.hpp"
#include "RiskAnalyzer.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

struct LanePosition {
    bool inside = false;
    float normalizedX = 0.5F;
};

struct ObjectGeometry {
    cv::Point groundPoint;
    bool insideRoad = false;
    bool insideEgoLane = false;
    float normalizedLaneX = 0.5F;
    float leadScore = -std::numeric_limits<float>::infinity();
};

/*
 * 편집된 테스트 영상의 장면 전환을 감지합니다.
 *
 * 이 프로젝트의 입력 영상은 하나의 연속 주행 영상이 아니라
 * 서로 다른 도로 장면이 이어 붙여진 영상입니다.
 *
 * 장면이 갑자기 바뀌었는데도 이전 장면의 Tracking ID,
 * TTC-P 이력, 경고 유지 카운터를 그대로 사용하면
 * 새 장면에 아무 차량이 없어도 이전 경고가 잠깐 남을 수 있습니다.
 *
 * 감지 방법:
 * 1. 프레임을 320×180 회색 영상으로 축소
 * 2. 이전 프레임과 평균 절대 차이(MAD) 계산
 * 3. 밝기 히스토그램 상관계수 계산
 * 4. 영상 전체 변화가 크고 히스토그램도 달라졌을 때 장면 전환으로 판단
 */
class SceneChangeDetector {
public:
    bool update(
        const cv::Mat& frame,
        float* meanAbsoluteDifference = nullptr,
        float* histogramCorrelation = nullptr
    ) {
        cv::Mat resized;
        cv::resize(
            frame,
            resized,
            cv::Size(320, 180)
        );

        cv::Mat gray;
        cv::cvtColor(
            resized,
            gray,
            cv::COLOR_BGR2GRAY
        );

        cv::Mat histogram;
        const int channels[] = {0};
        const int histogramSize[] = {64};
        const float range[] = {0.0F, 256.0F};
        const float* ranges[] = {range};

        cv::calcHist(
            &gray,
            1,
            channels,
            cv::Mat(),
            histogram,
            1,
            histogramSize,
            ranges,
            true,
            false
        );

        cv::normalize(
            histogram,
            histogram,
            1.0,
            0.0,
            cv::NORM_L1
        );

        /*
         * 첫 프레임에는 비교 대상이 없으므로
         * 현재 프레임만 저장하고 장면 전환은 아니라고 반환합니다.
         */
        if (
            previousGray_.empty() ||
            previousHistogram_.empty()
        ) {
            previousGray_ = gray.clone();
            previousHistogram_ = histogram.clone();

            if (meanAbsoluteDifference != nullptr) {
                *meanAbsoluteDifference = 0.0F;
            }

            if (histogramCorrelation != nullptr) {
                *histogramCorrelation = 1.0F;
            }

            return false;
        }

        cv::Mat difference;
        cv::absdiff(
            gray,
            previousGray_,
            difference
        );

        const float frameDifference =
            static_cast<float>(
                cv::mean(difference)[0]
            );

        const float correlation =
            static_cast<float>(
                cv::compareHist(
                    previousHistogram_,
                    histogram,
                    cv::HISTCMP_CORREL
                )
            );

        if (meanAbsoluteDifference != nullptr) {
            *meanAbsoluteDifference = frameDifference;
        }

        if (histogramCorrelation != nullptr) {
            *histogramCorrelation = correlation;
        }

        /*
         * 현재 결과 영상 분석에서 일반 주행 프레임 간 차이는
         * 대체로 1~4 수준이었고, 10.74초 장면 전환은 약 30.7이었습니다.
         *
         * 단순 카메라 흔들림을 장면 전환으로 잘못 보지 않도록
         * 프레임 차이와 히스토그램 상관도를 함께 확인합니다.
         */
        const bool sceneChanged =
            (
                frameDifference >= 20.0F &&
                correlation <= 0.78F
            ) ||
            frameDifference >= 40.0F;

        previousGray_ = gray.clone();
        previousHistogram_ = histogram.clone();

        return sceneChanged;
    }

private:
    cv::Mat previousGray_;
    cv::Mat previousHistogram_;
};

/*
 * OpenCV 기본 Hershey 글꼴은 한글을 지원하지 않습니다.
 * 따라서 opencv_contrib의 FreeType 모듈과 시스템에 설치된
 * 나눔고딕 또는 Noto Sans CJK 글꼴을 사용합니다.
 */
class KoreanTextRenderer {
public:
    bool initialize() {
        std::vector<std::string> fontCandidates;

        if (const char* customFont = std::getenv("ADAS_KOREAN_FONT")) {
            if (*customFont != '\0') {
                fontCandidates.emplace_back(customFont);
            }
        }

        fontCandidates.emplace_back(
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        );
        fontCandidates.emplace_back(
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
        );
        fontCandidates.emplace_back(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        );
        fontCandidates.emplace_back(
            "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf"
        );

        for (const std::string& fontPath : fontCandidates) {
            if (!std::filesystem::exists(fontPath)) {
                continue;
            }

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
    }

    void putText(
        cv::Mat& image,
        const std::string& text,
        const cv::Point& origin,
        int fontHeight,
        const cv::Scalar& color,
        int thickness = -1
    ) const {
        renderer_->putText(
            image,
            text,
            origin,
            fontHeight,
            color,
            thickness,
            cv::LINE_AA,
            false
        );
    }

    cv::Size getTextSize(
        const std::string& text,
        int fontHeight,
        int thickness,
        int* baseline
    ) const {
        return renderer_->getTextSize(
            text,
            fontHeight,
            thickness,
            baseline
        );
    }

    const std::string& fontPath() const {
        return fontPath_;
    }

private:
    cv::Ptr<cv::freetype::FreeType2> renderer_;
    std::string fontPath_;
};

bool isTargetClass(int classId) {
    return classId == 0 ||
           classId == 1 ||
           classId == 2 ||
           classId == 3 ||
           classId == 5 ||
           classId == 7;
}

bool isVehicleClass(int classId) {
    return classId == 2 ||
           classId == 3 ||
           classId == 5 ||
           classId == 7;
}

std::string getClassName(int classId) {
    switch (classId) {
        case 0: return "person";
        case 1: return "bicycle";
        case 2: return "car";
        case 3: return "motorcycle";
        case 5: return "bus";
        case 7: return "truck";
        default: return "unknown";
    }
}

std::string getRiskNameKorean(RiskLevel level) {
    switch (level) {
        case RiskLevel::Safe: return "안전";
        case RiskLevel::Caution: return "주의";
        case RiskLevel::Danger: return "위험";
    }

    return "알 수 없음";
}

int getNmsGroup(int classId) {
    if (classId == 2 || classId == 5 || classId == 7) {
        return 0;
    }

    if (classId == 1 || classId == 3) {
        return 1;
    }

    if (classId == 0) {
        return 2;
    }

    return classId + 10;
}

LetterboxResult letterbox(
    const cv::Mat& frame,
    int inputSize
) {
    const float scale = std::min(
        static_cast<float>(inputSize) /
            static_cast<float>(frame.cols),
        static_cast<float>(inputSize) /
            static_cast<float>(frame.rows)
    );

    const int resizedWidth = static_cast<int>(
        std::round(static_cast<float>(frame.cols) * scale)
    );

    const int resizedHeight = static_cast<int>(
        std::round(static_cast<float>(frame.rows) * scale)
    );

    cv::Mat resized;
    cv::resize(
        frame,
        resized,
        cv::Size(resizedWidth, resizedHeight)
    );

    const int totalPadX = inputSize - resizedWidth;
    const int totalPadY = inputSize - resizedHeight;

    const int padLeft = totalPadX / 2;
    const int padRight = totalPadX - padLeft;
    const int padTop = totalPadY / 2;
    const int padBottom = totalPadY - padTop;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    return {
        padded,
        scale,
        padLeft,
        padTop
    };
}

LanePosition calculateLanePosition(
    const std::vector<cv::Point>& trapezoid,
    const cv::Point& point
) {
    if (trapezoid.size() != 4) {
        return {};
    }

    const float topY =
        (
            static_cast<float>(trapezoid[0].y) +
            static_cast<float>(trapezoid[1].y)
        ) / 2.0F;

    const float bottomY =
        (
            static_cast<float>(trapezoid[2].y) +
            static_cast<float>(trapezoid[3].y)
        ) / 2.0F;

    if (
        point.y < topY ||
        point.y > bottomY ||
        bottomY <= topY
    ) {
        return {};
    }

    const float verticalRatio = std::clamp(
        (static_cast<float>(point.y) - topY) /
            (bottomY - topY),
        0.0F,
        1.0F
    );

    const float leftX =
        static_cast<float>(trapezoid[0].x) +
        (
            static_cast<float>(trapezoid[3].x) -
            static_cast<float>(trapezoid[0].x)
        ) * verticalRatio;

    const float rightX =
        static_cast<float>(trapezoid[1].x) +
        (
            static_cast<float>(trapezoid[2].x) -
            static_cast<float>(trapezoid[1].x)
        ) * verticalRatio;

    if (
        point.x < leftX ||
        point.x > rightX ||
        rightX <= leftX
    ) {
        return {};
    }

    return {
        true,
        (static_cast<float>(point.x) - leftX) /
            (rightX - leftX)
    };
}

/*
 * 우측 탱크로리 복구가 포함된 YOLO 검출 함수입니다.
 *
 * 진단 CSV에서 21초대 탱크로리는 박스 위치는 정상적으로 나오지만
 * 최고 클래스가 train(6)으로 출력되는 것이 확인됐습니다.
 * 이 함수는 해당 형태와 위치 조건을 만족하는 큰 우측 train 후보만
 * truck(7)으로 변경한 뒤 트래커에 전달합니다.
 */
std::vector<Detection> detectObjects(
    cv::dnn::Net& net,
    const cv::Mat& frame,
    float confidenceThreshold,
    float nmsThreshold
) {
    constexpr int inputSize = 640;

    const LetterboxResult prepared =
        letterbox(frame, inputSize);

    cv::Mat blob = cv::dnn::blobFromImage(
        prepared.image,
        1.0 / 255.0,
        cv::Size(inputSize, inputSize),
        cv::Scalar(),
        true,
        false
    );

    net.setInput(blob);
    cv::Mat output = net.forward();

    if (output.dims != 3) {
        throw std::runtime_error(
            "지원하지 않는 YOLO 출력 차원입니다."
        );
    }

    const int firstDimension = output.size[1];
    const int secondDimension = output.size[2];

    cv::Mat predictions(
        firstDimension,
        secondDimension,
        CV_32F,
        output.ptr<float>()
    );

    if (firstDimension < secondDimension) {
        predictions = predictions.t();
    }

    std::vector<cv::Rect> boxes;

    /*
     * confidences는 실제 화면 표시와 트래커 전달용 점수입니다.
     * nmsScores는 중복 박스 제거 우선순위에만 사용하는 점수입니다.
     *
     * 복구된 train 후보의 표시 점수는 최대 0.49로 제한하지만,
     * NMS에서는 원래 강한 객체 점수를 사용해 작은 car 박스에
     * 큰 탱크로리 박스가 밀리지 않게 합니다.
     */
    std::vector<float> confidences;
    std::vector<float> nmsScores;
    std::vector<int> classIds;

    for (int row = 0; row < predictions.rows; ++row) {
        const float* data = predictions.ptr<float>(row);

        const float centerX = data[0];
        const float centerY = data[1];
        const float boxWidth = data[2];
        const float boxHeight = data[3];

        cv::Mat classScores(
            1,
            predictions.cols - 4,
            CV_32F,
            const_cast<float*>(data + 4)
        );

        cv::Point bestClassPoint;
        double bestClassScore = 0.0;

        cv::minMaxLoc(
            classScores,
            nullptr,
            &bestClassScore,
            nullptr,
            &bestClassPoint
        );

        const int rawClassId = bestClassPoint.x;
        const float rawConfidence =
            static_cast<float>(bestClassScore);

        if (rawConfidence < confidenceThreshold) {
            continue;
        }

        int left = static_cast<int>(std::round(
            (
                centerX -
                boxWidth / 2.0F -
                static_cast<float>(prepared.padX)
            ) / prepared.scale
        ));

        int top = static_cast<int>(std::round(
            (
                centerY -
                boxHeight / 2.0F -
                static_cast<float>(prepared.padY)
            ) / prepared.scale
        ));

        int right = static_cast<int>(std::round(
            (
                centerX +
                boxWidth / 2.0F -
                static_cast<float>(prepared.padX)
            ) / prepared.scale
        ));

        int bottom = static_cast<int>(std::round(
            (
                centerY +
                boxHeight / 2.0F -
                static_cast<float>(prepared.padY)
            ) / prepared.scale
        ));

        left = std::clamp(left, 0, frame.cols - 1);
        top = std::clamp(top, 0, frame.rows - 1);
        right = std::clamp(right, 0, frame.cols - 1);
        bottom = std::clamp(bottom, 0, frame.rows - 1);

        if (right <= left || bottom <= top) {
            continue;
        }

        const int restoredWidth = right - left;
        const int restoredHeight = bottom - top;
        const int restoredCenterX = left + restoredWidth / 2;
        const int restoredArea = restoredWidth * restoredHeight;

        /*
         * 21초대 우측 탱크로리 train -> truck 복구 조건입니다.
         *
         * 실제 진단 박스보다 약간 여유 있게 설정해
         * 프레임별 박스 흔들림 때문에 조건이 끊기는 것을 줄였습니다.
         * 모든 train을 바꾸는 것이 아니라 우측의 큰 도로 객체만 복구합니다.
         */
        const bool isRightTankerMisclassifiedAsTrain =
            rawClassId == 6 &&
            rawConfidence >= 0.20F &&
            left >= static_cast<int>(frame.cols * 0.45F) &&
            restoredCenterX >=
                static_cast<int>(frame.cols * 0.60F) &&
            right >= static_cast<int>(frame.cols * 0.76F) &&
            restoredWidth >=
                static_cast<int>(frame.cols * 0.18F) &&
            restoredHeight >=
                static_cast<int>(frame.rows * 0.20F) &&
            restoredArea >= static_cast<int>(
                static_cast<float>(frame.cols * frame.rows) * 0.055F
            ) &&
            bottom >= static_cast<int>(frame.rows * 0.62F);

        const int finalClassId =
            isRightTankerMisclassifiedAsTrain
                ? 7
                : rawClassId;

        if (!isTargetClass(finalClassId)) {
            continue;
        }

        const float displayConfidence =
            isRightTankerMisclassifiedAsTrain
                ? std::clamp(
                      rawConfidence * 0.55F,
                      0.25F,
                      0.49F
                  )
                : rawConfidence;

        boxes.emplace_back(
            left,
            top,
            restoredWidth,
            restoredHeight
        );

        confidences.push_back(displayConfidence);
        nmsScores.push_back(rawConfidence);
        classIds.push_back(finalClassId);
    }

    std::map<int, std::vector<int>> indicesByGroup;

    for (int index = 0;
         index < static_cast<int>(classIds.size());
         ++index) {
        indicesByGroup[
            getNmsGroup(classIds[index])
        ].push_back(index);
    }

    std::vector<Detection> detections;

    for (const auto& groupEntry : indicesByGroup) {
        const std::vector<int>& globalIndices =
            groupEntry.second;

        std::vector<cv::Rect> groupBoxes;
        std::vector<float> groupNmsScores;

        for (const int globalIndex : globalIndices) {
            groupBoxes.push_back(boxes[globalIndex]);
            groupNmsScores.push_back(nmsScores[globalIndex]);
        }

        std::vector<int> keptIndices;

        cv::dnn::NMSBoxes(
            groupBoxes,
            groupNmsScores,
            confidenceThreshold,
            nmsThreshold,
            keptIndices
        );

        for (const int localIndex : keptIndices) {
            const int globalIndex =
                globalIndices[localIndex];

            detections.push_back({
                classIds[globalIndex],
                confidences[globalIndex],
                boxes[globalIndex]
            });
        }
    }

    return detections;
}

void drawCenteredKoreanText(
    KoreanTextRenderer& renderer,
    cv::Mat& frame,
    const std::string& text,
    int y,
    int fontHeight,
    const cv::Scalar& color
) {
    int baseline = 0;

    const cv::Size textSize = renderer.getTextSize(
        text,
        fontHeight,
        -1,
        &baseline
    );

    const int x = std::max(
        0,
        (frame.cols - textSize.width) / 2
    );

    renderer.putText(
        frame,
        text,
        cv::Point(x, y),
        fontHeight,
        color,
        -1
    );
}

int main(int argc, char* argv[]) {
    const std::string inputPath =
        argc >= 2
            ? argv[1]
            : "videos/input.mp4";

    const std::string modelPath =
        "models/yolov8n.onnx";

    const std::string outputPath =
        "results/step7_scene_change_fixed_output.avi";

    std::filesystem::create_directories("results");

    if (!std::filesystem::exists(modelPath)) {
        std::cerr
            << "[ERROR] YOLO 모델을 찾을 수 없습니다: "
            << modelPath
            << '\n';
        return 1;
    }

    KoreanTextRenderer koreanText;

    if (!koreanText.initialize()) {
        std::cerr
            << "[ERROR] 한글 글꼴을 찾거나 불러오지 못했습니다.\n"
            << "다음 명령으로 필요한 패키지를 설치하세요:\n"
            << "sudo apt install -y libopencv-contrib-dev fonts-nanum\n";
        return 1;
    }

    std::cout
        << "[SUCCESS] 한글 글꼴: "
        << koreanText.fontPath()
        << '\n';

    cv::dnn::Net net;

    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );
        net.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );
    } catch (const cv::Exception& error) {
        std::cerr
            << "[ERROR] YOLO 모델 로드 실패\n"
            << error.what()
            << '\n';
        return 1;
    }

    cv::VideoCapture capture(inputPath);

    if (!capture.isOpened()) {
        std::cerr
            << "[ERROR] 영상을 열 수 없습니다: "
            << inputPath
            << '\n';
        return 1;
    }

    const int width = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_WIDTH)
    );

    const int height = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    double sourceFps =
        capture.get(cv::CAP_PROP_FPS);

    if (sourceFps <= 0.0) {
        sourceFps = 30.0;
    }

    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        sourceFps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr
            << "[ERROR] 결과 영상을 만들 수 없습니다: "
            << outputPath
            << '\n';
        return 1;
    }

    /*
     * 넓은 도로 관심 영역입니다.
     * 실제 위험 판단은 아래 egoLaneRoi의 선행 차량 한 대에만 적용합니다.
     */
    const std::vector<cv::Point> roadRoi = {
        cv::Point(
            static_cast<int>(width * 0.37F),
            static_cast<int>(height * 0.64F)
        ),
        cv::Point(
            static_cast<int>(width * 0.64F),
            static_cast<int>(height * 0.64F)
        ),
        cv::Point(
            static_cast<int>(width * 0.96F),
            static_cast<int>(height * 0.96F)
        ),
        cv::Point(
            static_cast<int>(width * 0.04F),
            static_cast<int>(height * 0.96F)
        )
    };

    /*
     * 현재 테스트 영상에서 내 차량이 주행하는 차선 ROI입니다.
     */
    const std::vector<cv::Point> egoLaneRoi = {
        cv::Point(
            static_cast<int>(width * 0.40F),
            static_cast<int>(height * 0.65F)
        ),
        cv::Point(
            static_cast<int>(width * 0.48F),
            static_cast<int>(height * 0.65F)
        ),
        cv::Point(
            static_cast<int>(width * 0.68F),
            static_cast<int>(height * 0.95F)
        ),
        cv::Point(
            static_cast<int>(width * 0.10F),
            static_cast<int>(height * 0.95F)
        )
    };

    constexpr float detectorThreshold = 0.10F;
    constexpr float nmsThreshold = 0.45F;

    MultiObjectTracker tracker(
        0.25F,
        0.10F,
        3,
        20
    );

    RiskAnalyzer riskAnalyzer(
        sourceFps,
        15,
        60
    );

    int activeLeadId = -1;

    /*
     * 경고가 한두 프레임 만에 사라져 깜빡이는 현상을 줄이기 위한
     * 상단 경고 유지 시간입니다.
     *
     * 기존 0.4초보다 조금 줄여 정상 상태로 돌아온 뒤
     * 오래 남아 보이지 않도록 0.25초로 조정했습니다.
     */
    const int warningHoldFrames = std::max(
        1,
        static_cast<int>(
            std::round(sourceFps * 0.25)
        )
    );

    /*
     * TTC-P는 바운딩 박스 높이 변화로 계산되므로,
     * 멀리 있는 아주 작은 차량은 박스가 2~3픽셀만 흔들려도
     * 급격히 접근하는 것처럼 계산될 수 있습니다.
     *
     * 따라서 상단 경고와 CAUTION/DANGER 표시는
     * 아래 신뢰성 조건을 모두 만족할 때만 허용합니다.
     */
    const int minimumLeadBoxHeightForWarning =
        std::max(
            36,
            static_cast<int>(
                std::round(height * 0.05)
            )
        );

    const int minimumLeadGroundYForWarning =
        static_cast<int>(
            std::round(height * 0.60)
        );

    constexpr int minimumRiskSamplesForWarning = 10;
    constexpr float minimumHeightGrowthRatioForWarning = 1.08F;
    constexpr float minimumGroundSpeedForWarning = 2.0F;

    int cautionHoldRemaining = 0;
    int dangerHoldRemaining = 0;

    /*
     * RiskAnalyzer 자체의 안정화 외에도,
     * 상단의 큰 경고 배너는 같은 선행 차량에서 위험 신호가
     * 일정 프레임 이상 연속될 때만 표시합니다.
     *
     * 10.57초부터 장면이 바뀌기 전까지는 주의 상태가 약 5프레임만
     * 나타났으므로, 8프레임 연속 확인 조건을 적용하면
     * 이런 짧은 오경고가 상단에 뜨지 않습니다.
     */
    constexpr int cautionConfirmationFrames = 8;
    constexpr int dangerConfirmationFrames = 3;

    int cautionCandidateFrames = 0;
    int dangerCandidateFrames = 0;
    int warningCandidateLeadId = -1;

    /*
     * 장면 전환 직후에는 이전 장면의 추적과 TTC-P 이력이
     * 새 장면으로 이어지지 않도록 약 0.5초 동안 위험 분석을 쉽니다.
     *
     * 이 시간 동안 RiskAnalyzer에는 모든 객체를
     * isLeadTarget=false로 전달해 기존 ID의 기록도 초기화합니다.
     */
    const int sceneWarmupFrames = std::max(
        1,
        static_cast<int>(
            std::round(sourceFps * 0.5)
        )
    );

    int sceneWarmupRemaining = 0;

    SceneChangeDetector sceneChangeDetector;

    cv::Mat frame;
    int processedFrames = 0;
    double totalInferenceMilliseconds = 0.0;

    const auto totalStart =
        std::chrono::steady_clock::now();

    while (capture.read(frame)) {
        ++processedFrames;

        /*
         * 반드시 ROI, 박스, 글씨를 그리기 전의 원본 프레임으로
         * 장면 전환을 감지합니다.
         */
        float sceneDifference = 0.0F;
        float sceneHistogramCorrelation = 1.0F;

        const bool sceneChanged =
            sceneChangeDetector.update(
                frame,
                &sceneDifference,
                &sceneHistogramCorrelation
            );

        if (sceneChanged) {
            /*
             * 이전 장면에서 유지 중이던 경고를 즉시 제거합니다.
             */
            cautionHoldRemaining = 0;
            dangerHoldRemaining = 0;

            cautionCandidateFrames = 0;
            dangerCandidateFrames = 0;
            warningCandidateLeadId = -1;

            activeLeadId = -1;
            sceneWarmupRemaining =
                sceneWarmupFrames;

            std::cout
                << "[SCENE CHANGE] frame="
                << processedFrames
                << " diff="
                << std::fixed
                << std::setprecision(2)
                << sceneDifference
                << " histogram="
                << sceneHistogramCorrelation
                << '\n';
        }

        const bool riskAnalysisEnabled =
            sceneWarmupRemaining <= 0;

        const auto inferenceStart =
            std::chrono::steady_clock::now();

        std::vector<Detection> detections;

        try {
            detections = detectObjects(
                net,
                frame,
                detectorThreshold,
                nmsThreshold
            );
        } catch (const std::exception& error) {
            std::cerr
                << "[ERROR] 객체 검출 실패: "
                << error.what()
                << '\n';
            return 1;
        }

        const auto inferenceEnd =
            std::chrono::steady_clock::now();

        const double inferenceMilliseconds =
            std::chrono::duration<double, std::milli>(
                inferenceEnd - inferenceStart
            ).count();

        totalInferenceMilliseconds +=
            inferenceMilliseconds;

        const std::vector<TrackedObject> trackedObjects =
            tracker.update(detections);

        std::unordered_map<int, ObjectGeometry>
            geometryById;

        std::unordered_map<int, float>
            leadScoreById;

        int proposedLeadId = -1;

        float proposedLeadScore =
            -std::numeric_limits<float>::infinity();

        /*
         * 먼저 모든 객체의 접지점과 차선 위치를 계산하고
         * 내 차선에서 가장 가까운 선행 차량 후보를 선택합니다.
         */
        for (const TrackedObject& trackedObject
             : trackedObjects) {
            const cv::Rect& box = trackedObject.box;

            const cv::Point groundPoint(
                box.x + box.width / 2,
                box.y + box.height
            );

            const bool insideRoad =
                cv::pointPolygonTest(
                    roadRoi,
                    groundPoint,
                    false
                ) >= 0.0;

            const LanePosition lanePosition =
                calculateLanePosition(
                    egoLaneRoi,
                    groundPoint
                );

            float leadScore =
                -std::numeric_limits<float>::infinity();

            if (
                lanePosition.inside &&
                isVehicleClass(trackedObject.classId)
            ) {
                const float centerPenalty =
                    std::abs(
                        lanePosition.normalizedX - 0.5F
                    ) * 90.0F;

                leadScore =
                    static_cast<float>(groundPoint.y) -
                    centerPenalty;

                leadScoreById[
                    trackedObject.trackId
                ] = leadScore;

                if (leadScore > proposedLeadScore) {
                    proposedLeadScore = leadScore;
                    proposedLeadId =
                        trackedObject.trackId;
                }
            }

            geometryById[
                trackedObject.trackId
            ] = {
                groundPoint,
                insideRoad,
                lanePosition.inside,
                lanePosition.normalizedX,
                leadScore
            };
        }

        const auto activeLeadIterator =
            leadScoreById.find(activeLeadId);

        const bool activeLeadVisible =
            activeLeadIterator != leadScoreById.end();

        /*
         * 장면 전환 직후에는 선행 차량을 바로 선택하지 않습니다.
         * 새 장면에서 추적 정보가 다시 안정화된 뒤에만 선택합니다.
         */
        if (!riskAnalysisEnabled) {
            activeLeadId = -1;
        } else if (!activeLeadVisible) {
            activeLeadId = proposedLeadId;
        } else if (
            proposedLeadId >= 0 &&
            proposedLeadId != activeLeadId &&
            proposedLeadScore >
                activeLeadIterator->second + 35.0F
        ) {
            activeLeadId = proposedLeadId;
        }

        /*
         * 검출은 아무것도 그리지 않은 원본 프레임에서 끝났습니다.
         * 그 뒤에만 ROI를 표시하므로 오버레이가 YOLO 입력을 훼손하지 않습니다.
         *
         * 진단용 자홍색 영역과 RAW YOLO 후보는 이 최종 버전에서
         * 완전히 제거했습니다.
         */
        cv::Mat overlay = frame.clone();

        cv::fillConvexPoly(
            overlay,
            roadRoi,
            cv::Scalar(0, 255, 0)
        );

        cv::addWeighted(
            overlay,
            0.10,
            frame,
            0.90,
            0.0,
            frame
        );

        cv::polylines(
            frame,
            roadRoi,
            true,
            cv::Scalar(0, 180, 0),
            2
        );

        cv::polylines(
            frame,
            egoLaneRoi,
            true,
            cv::Scalar(255, 255, 255),
            3
        );

        int objectsOnRoad = 0;
        int objectsInEgoLane = 0;

        RiskResult leadRisk;
        bool leadRiskFound = false;

        for (const TrackedObject& trackedObject
             : trackedObjects) {
            const auto geometryIterator =
                geometryById.find(
                    trackedObject.trackId
                );

            if (geometryIterator == geometryById.end()) {
                continue;
            }

            const ObjectGeometry& geometry =
                geometryIterator->second;

            if (geometry.insideRoad) {
                ++objectsOnRoad;
            }

            if (geometry.insideEgoLane) {
                ++objectsInEgoLane;
            }

            const bool isLeadTarget =
                riskAnalysisEnabled &&
                trackedObject.trackId == activeLeadId &&
                geometry.insideEgoLane;

            const RiskResult rawRisk =
                riskAnalyzer.update(
                    trackedObject,
                    isLeadTarget,
                    processedFrames
                );

            /*
             * 원시 TTC-P 결과를 그대로 경고에 사용하지 않고
             * 화면상 거리와 접근 움직임을 함께 확인합니다.
             *
             * 14.98~15.58초의 오경고는 멀리 있는 작은 선행 차량의
             * 박스 높이가 몇 픽셀 흔들리면서 TTC-P가 약 4초로
             * 잘못 계산되어 발생했습니다.
             *
             * 다음을 모두 만족해야 CAUTION/DANGER를 인정합니다.
             *
             * 1. TTC-P 계산이 유효함
             * 2. 관찰 샘플이 충분함
             * 3. 선행 차량 박스가 너무 작지 않음
             * 4. 차량 접지점이 화면에서 충분히 가까운 위치에 있음
             * 5. 박스 높이가 전체 관찰 구간에서 의미 있게 증가함
             * 6. 접지점도 아래 방향으로 이동하고 있음
             */
            const bool hasReliableWarningEvidence =
                isLeadTarget &&
                rawRisk.valid &&
                std::isfinite(rawRisk.ttcSeconds) &&
                rawRisk.sampleCount >=
                    minimumRiskSamplesForWarning &&
                trackedObject.box.height >=
                    minimumLeadBoxHeightForWarning &&
                geometry.groundPoint.y >=
                    minimumLeadGroundYForWarning &&
                rawRisk.heightGrowthRatio >=
                    minimumHeightGrowthRatioForWarning &&
                rawRisk.groundSpeedPixelsPerSecond >=
                    minimumGroundSpeedForWarning;

            /*
             * 내부 RiskAnalyzer의 계산 결과는 유지하되,
             * 신뢰성 조건을 통과하지 못한 CAUTION/DANGER는
             * 화면 표시와 상단 경고에서는 SAFE로 처리합니다.
             */
            RiskResult risk = rawRisk;

            if (
                risk.level != RiskLevel::Safe &&
                !hasReliableWarningEvidence
            ) {
                risk.level = RiskLevel::Safe;
                risk.valid = false;
                risk.ttcSeconds =
                    std::numeric_limits<float>::infinity();
            }

            if (isLeadTarget) {
                leadRisk = risk;
                leadRiskFound = true;
            }

            cv::Scalar boxColor;

            if (!geometry.insideRoad) {
                boxColor = cv::Scalar(0, 165, 255);
            } else if (!geometry.insideEgoLane) {
                boxColor = cv::Scalar(255, 255, 0);
            } else if (!isLeadTarget) {
                boxColor = cv::Scalar(255, 180, 0);
            } else if (risk.level == RiskLevel::Danger) {
                boxColor = cv::Scalar(0, 0, 255);
            } else if (risk.level == RiskLevel::Caution) {
                boxColor = cv::Scalar(0, 255, 255);
            } else {
                boxColor = cv::Scalar(0, 255, 0);
            }

            cv::rectangle(
                frame,
                trackedObject.box,
                boxColor,
                3
            );

            cv::circle(
                frame,
                geometry.groundPoint,
                6,
                boxColor,
                -1
            );

            /*
             * 객체 클래스명과 박스 라벨은 요청대로 영어를 유지합니다.
             */
            std::string label =
                getClassName(trackedObject.classId) +
                " ID:" +
                std::to_string(trackedObject.trackId) +
                " " +
                cv::format(
                    "%.2f",
                    trackedObject.confidence
                );

            if (isLeadTarget) {
                label =
                    "LEAD " +
                    label +
                    " " +
                    RiskAnalyzer::toString(risk.level);

                if (
                    risk.valid &&
                    std::isfinite(risk.ttcSeconds)
                ) {
                    label += cv::format(
                        " TTC-P:%.1fs",
                        risk.ttcSeconds
                    );
                } else {
                    label += " TTC-P:--";
                }
            } else if (geometry.insideEgoLane) {
                label += " EGO-LANE";
            } else if (geometry.insideRoad) {
                label += " SIDE";
            }

            int baseline = 0;

            const cv::Size labelSize =
                cv::getTextSize(
                    label,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2,
                    &baseline
                );

            const int labelX = std::clamp(
                trackedObject.box.x,
                0,
                std::max(
                    width - labelSize.width - 10,
                    0
                )
            );

            const int labelTop = std::max(
                trackedObject.box.y,
                labelSize.height + 10
            );

            cv::rectangle(
                frame,
                cv::Point(
                    labelX,
                    labelTop - labelSize.height - 10
                ),
                cv::Point(
                    labelX + labelSize.width + 10,
                    labelTop
                ),
                boxColor,
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(labelX + 5, labelTop - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 0),
                2,
                cv::LINE_AA
            );
        }

        riskAnalyzer.removeStaleTracks(processedFrames);

        /*
         * 상단 경고 배너는 다음 세 단계를 모두 통과해야 표시됩니다.
         *
         * 1. 장면 전환 직후의 분석 대기 시간이 아닐 것
         * 2. 같은 선행 차량에서 위험 상태가 연속으로 관찰될 것
         * 3. CAUTION은 8프레임, DANGER는 3프레임 이상 지속될 것
         *
         * 장면이 바뀌면 이전 경고 유지 카운터를 즉시 0으로 만들기 때문에,
         * 새 화면에 차량이 없는데 이전 장면의 경고가 남는 현상이 사라집니다.
         */
        if (!riskAnalysisEnabled || sceneChanged) {
            cautionHoldRemaining = 0;
            dangerHoldRemaining = 0;

            cautionCandidateFrames = 0;
            dangerCandidateFrames = 0;
            warningCandidateLeadId = -1;
        } else if (
            leadRiskFound &&
            activeLeadId >= 0
        ) {
            /*
             * 선행 차량 ID가 바뀌면 이전 차량에서 쌓인
             * 위험 연속 프레임 수를 이어받지 않습니다.
             */
            if (warningCandidateLeadId != activeLeadId) {
                warningCandidateLeadId =
                    activeLeadId;

                cautionCandidateFrames = 0;
                dangerCandidateFrames = 0;
            }

            if (leadRisk.level == RiskLevel::Danger) {
                ++dangerCandidateFrames;
                cautionCandidateFrames = 0;

                if (
                    dangerCandidateFrames >=
                    dangerConfirmationFrames
                ) {
                    dangerHoldRemaining =
                        warningHoldFrames;

                    cautionHoldRemaining = 0;
                }
            } else if (
                leadRisk.level == RiskLevel::Caution
            ) {
                ++cautionCandidateFrames;
                dangerCandidateFrames = 0;

                if (
                    cautionCandidateFrames >=
                    cautionConfirmationFrames
                ) {
                    cautionHoldRemaining =
                        warningHoldFrames;
                }

                if (dangerHoldRemaining > 0) {
                    --dangerHoldRemaining;
                }
            } else {
                cautionCandidateFrames = 0;
                dangerCandidateFrames = 0;

                if (dangerHoldRemaining > 0) {
                    --dangerHoldRemaining;
                }

                if (cautionHoldRemaining > 0) {
                    --cautionHoldRemaining;
                }
            }
        } else {
            warningCandidateLeadId = -1;
            cautionCandidateFrames = 0;
            dangerCandidateFrames = 0;

            if (dangerHoldRemaining > 0) {
                --dangerHoldRemaining;
            }

            if (cautionHoldRemaining > 0) {
                --cautionHoldRemaining;
            }
        }

        /*
         * 장면 전환 직후의 위험 분석 대기 시간을 한 프레임 줄입니다.
         */
        if (sceneWarmupRemaining > 0) {
            --sceneWarmupRemaining;
        }

        /*
         * 왼쪽 위 상태 정보는 요청대로 한국어로 표시합니다.
         */
        koreanText.putText(
            frame,
            "프레임: " + std::to_string(processedFrames),
            cv::Point(30, 40),
            25,
            cv::Scalar(255, 255, 255)
        );

        koreanText.putText(
            frame,
            "추론 시간: " +
                std::string(cv::format(
                    "%.1f ms",
                    inferenceMilliseconds
                )),
            cv::Point(30, 72),
            25,
            cv::Scalar(255, 255, 255)
        );

        koreanText.putText(
            frame,
            "도로 객체: " +
                std::to_string(objectsOnRoad) +
                "  내 차선: " +
                std::to_string(objectsInEgoLane),
            cv::Point(30, 104),
            25,
            cv::Scalar(255, 255, 255)
        );

        std::string leadStatus =
            "선행 차량: 없음";

        if (
            activeLeadId >= 0 &&
            leadRiskFound
        ) {
            leadStatus =
                "선행 차량 ID:" +
                std::to_string(activeLeadId) +
                "  상태: " +
                getRiskNameKorean(leadRisk.level);

            if (
                leadRisk.valid &&
                std::isfinite(leadRisk.ttcSeconds)
            ) {
                leadStatus +=
                    "  TTC-P:" +
                    std::string(cv::format(
                        "%.1f초",
                        leadRisk.ttcSeconds
                    ));
            } else {
                leadStatus += "  TTC-P:--";
            }
        }

        koreanText.putText(
            frame,
            leadStatus,
            cv::Point(30, 136),
            25,
            cv::Scalar(255, 255, 255)
        );

        /*
         * 화면 상단 경고도 요청대로 한국어로 표시합니다.
         */
        if (dangerHoldRemaining > 0) {
            drawCenteredKoreanText(
                koreanText,
                frame,
                "위험: 충돌 가능성이 높습니다",
                58,
                34,
                cv::Scalar(0, 0, 255)
            );
        } else if (cautionHoldRemaining > 0) {
            drawCenteredKoreanText(
                koreanText,
                frame,
                "주의: 선행 차량에 접근 중입니다",
                58,
                31,
                cv::Scalar(0, 255, 255)
            );
        }

        /*
         * 기존 왼쪽 아래의
         * "TTC-P = image-scale based proxy"
         * 문구는 요청대로 완전히 제거했습니다.
         */

        writer.write(frame);
    }

    const auto totalEnd =
        std::chrono::steady_clock::now();

    const double elapsedSeconds =
        std::chrono::duration<double>(
            totalEnd - totalStart
        ).count();

    const double processingFps =
        elapsedSeconds > 0.0
            ? static_cast<double>(processedFrames) /
                elapsedSeconds
            : 0.0;

    const double averageInferenceMilliseconds =
        processedFrames > 0
            ? totalInferenceMilliseconds /
                static_cast<double>(processedFrames)
            : 0.0;

    capture.release();
    writer.release();

    std::cout << '\n';
    std::cout
        << "[SUCCESS] Step 7 장면 전환 오경고 제거 영상 생성 완료\n";
    std::cout
        << "처리 프레임 수: "
        << processedFrames
        << '\n';
    std::cout
        << "평균 추론 시간: "
        << std::fixed
        << std::setprecision(2)
        << averageInferenceMilliseconds
        << " ms\n";
    std::cout
        << "전체 처리 속도: "
        << std::fixed
        << std::setprecision(2)
        << processingFps
        << " FPS\n";
    std::cout
        << "결과 파일: "
        << outputPath
        << '\n';

    return 0;
}