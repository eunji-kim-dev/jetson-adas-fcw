#include "Detection.hpp"
#include "MultiObjectTracker.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * Letterbox 전처리 결과입니다.
 *
 * image:
 * YOLO 입력 크기인 640 × 640으로 변환된 이미지
 *
 * scale:
 * 원본 영상을 축소하거나 확대할 때 적용한 비율
 *
 * padX, padY:
 * 가로와 세로에 추가한 패딩 위치
 *
 * YOLO가 출력한 640 × 640 기준 좌표를
 * 원본 영상 좌표로 복원할 때 사용합니다.
 */
struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

/*
 * COCO 데이터셋 80개 클래스 중
 * ADAS에서 사용할 객체만 통과시킵니다.
 *
 * 0: person
 * 1: bicycle
 * 2: car
 * 3: motorcycle
 * 5: bus
 * 7: truck
 */
bool isTargetClass(int classId) {
    return classId == 0 ||
           classId == 1 ||
           classId == 2 ||
           classId == 3 ||
           classId == 5 ||
           classId == 7;
}

/*
 * COCO 클래스 번호를 화면에 표시할 이름으로 변환합니다.
 */
std::string getClassName(int classId) {
    switch (classId) {
        case 0:
            return "person";

        case 1:
            return "bicycle";

        case 2:
            return "car";

        case 3:
            return "motorcycle";

        case 5:
            return "bus";

        case 7:
            return "truck";

        default:
            return "unknown";
    }
}

/*
 * 추적기에서 동일한 종류로 취급하는 객체를
 * NMS 단계에서도 같은 그룹으로 묶습니다.
 *
 * 그룹 0:
 * car, bus, truck
 *
 * 그룹 1:
 * bicycle, motorcycle
 *
 * 그룹 2:
 * person
 *
 * 같은 차량이 car와 truck으로 동시에 검출됐을 때
 * 클래스별 NMS를 사용하면 두 박스가 모두 남을 수 있습니다.
 *
 * 그룹 NMS를 사용하면 서로 크게 겹치는 차량 박스 중
 * 신뢰도가 높은 하나만 남길 수 있습니다.
 */
int getNmsGroup(int classId) {
    // 사륜 차량 그룹
    if (
        classId == 2 ||
        classId == 5 ||
        classId == 7
    ) {
        return 0;
    }

    // 이륜 차량 그룹
    if (
        classId == 1 ||
        classId == 3
    ) {
        return 1;
    }

    // 보행자 그룹
    if (classId == 0) {
        return 2;
    }

    /*
     * 현재는 target class만 들어오므로
     * 사실상 실행되지 않는 예비 처리입니다.
     */
    return classId + 10;
}

/*
 * 원본 영상의 가로세로 비율을 유지하면서
 * YOLO 입력 크기인 정사각형 이미지로 변환합니다.
 *
 * 단순 resize를 사용하면 차량이 가로 또는 세로로
 * 찌그러져 검출 성능이 떨어질 수 있습니다.
 */
LetterboxResult letterbox(
    const cv::Mat& frame,
    int inputSize
) {
    /*
     * 가로와 세로 배율 중 더 작은 값을 사용합니다.
     *
     * 그래야 원본 영상 전체가 640 × 640 영역 안에
     * 잘리지 않고 들어갑니다.
     */
    const float scale = std::min(
        static_cast<float>(inputSize) /
            static_cast<float>(frame.cols),

        static_cast<float>(inputSize) /
            static_cast<float>(frame.rows)
    );

    const int resizedWidth =
        static_cast<int>(
            std::round(
                static_cast<float>(frame.cols) *
                scale
            )
        );

    const int resizedHeight =
        static_cast<int>(
            std::round(
                static_cast<float>(frame.rows) *
                scale
            )
        );

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(
            resizedWidth,
            resizedHeight
        )
    );

    /*
     * resize 후 남는 가로와 세로 공간을 계산합니다.
     */
    const int totalPadX =
        inputSize - resizedWidth;

    const int totalPadY =
        inputSize - resizedHeight;

    /*
     * 패딩을 양쪽에 나눠 넣어 영상을 중앙에 배치합니다.
     */
    const int padLeft =
        totalPadX / 2;

    const int padRight =
        totalPadX - padLeft;

    const int padTop =
        totalPadY / 2;

    const int padBottom =
        totalPadY - padTop;

    cv::Mat padded;

    /*
     * YOLO 계열 모델에서 일반적으로 사용하는
     * 회색 값 114로 패딩합니다.
     */
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

/*
 * 현재 프레임에서 YOLO 객체 검출을 수행합니다.
 *
 * detectorThreshold를 0.10으로 전달하므로
 * 고신뢰도 검출뿐 아니라 저신뢰도 검출도 반환합니다.
 *
 * 반환된 Detection은 MultiObjectTracker에서 다시:
 *
 * 0.25 이상
 * → 고신뢰도 검출
 *
 * 0.10 이상 0.25 미만
 * → 기존 Track 유지용 저신뢰도 검출
 *
 * 로 분리합니다.
 */
std::vector<Detection> detectObjects(
    cv::dnn::Net& net,
    const cv::Mat& frame,
    float confidenceThreshold,
    float nmsThreshold
) {
    constexpr int inputSize = 640;

    /*
     * 원본 비율을 유지한 640 × 640 입력을 만듭니다.
     */
    const LetterboxResult prepared =
        letterbox(
            frame,
            inputSize
        );

    /*
     * OpenCV 이미지를 YOLO 입력 텐서로 변환합니다.
     *
     * 1.0 / 255.0:
     * 픽셀값을 0~255에서 0~1 범위로 정규화
     *
     * swapRB = true:
     * OpenCV의 BGR 순서를 YOLO의 RGB 순서로 변경
     */
    cv::Mat blob =
        cv::dnn::blobFromImage(
            prepared.image,
            1.0 / 255.0,
            cv::Size(
                inputSize,
                inputSize
            ),
            cv::Scalar(),
            true,
            false
        );

    net.setInput(blob);

    /*
     * 실제 YOLO 추론을 수행합니다.
     */
    cv::Mat output =
        net.forward();

    if (output.dims != 3) {
        throw std::runtime_error(
            "지원하지 않는 YOLO 출력 차원입니다."
        );
    }

    /*
     * YOLOv8 ONNX 출력은 환경에 따라 다음 중 하나입니다.
     *
     * [1, 84, 8400]
     * [1, 8400, 84]
     *
     * 각 행이 검출 후보 하나가 되도록
     * 최종 형태를 [8400, 84]로 통일합니다.
     */
    const int firstDimension =
        output.size[1];

    const int secondDimension =
        output.size[2];

    cv::Mat predictions(
        firstDimension,
        secondDimension,
        CV_32F,
        output.ptr<float>()
    );

    if (firstDimension < secondDimension) {
        predictions =
            predictions.t();
    }

    /*
     * NMS 적용 전의 검출 후보를 저장합니다.
     */
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (
        int row = 0;
        row < predictions.rows;
        ++row
    ) {
        const float* data =
            predictions.ptr<float>(row);

        /*
         * YOLOv8 출력 앞의 네 값:
         *
         * 박스 중심 x
         * 박스 중심 y
         * 박스 너비
         * 박스 높이
         */
        const float centerX =
            data[0];

        const float centerY =
            data[1];

        const float boxWidth =
            data[2];

        const float boxHeight =
            data[3];

        /*
         * 4번 이후는 80개 클래스의 신뢰도입니다.
         */
        cv::Mat classScores(
            1,
            predictions.cols - 4,
            CV_32F,
            const_cast<float*>(
                data + 4
            )
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

        const int classId =
            bestClassPoint.x;

        const float confidence =
            static_cast<float>(
                bestClassScore
            );

        /*
         * 0.10 미만 검출은 추적에도 사용하지 않으므로 제거합니다.
         */
        if (
            confidence <
            confidenceThreshold
        ) {
            continue;
        }

        /*
         * ADAS 대상 클래스가 아니면 제거합니다.
         */
        if (!isTargetClass(classId)) {
            continue;
        }

        /*
         * 640 × 640 Letterbox 좌표를
         * 원본 영상 좌표로 복원합니다.
         *
         * 1. 패딩 제거
         * 2. resize 배율로 나누기
         */
        int left =
            static_cast<int>(
                std::round(
                    (
                        centerX -
                        boxWidth / 2.0F -
                        static_cast<float>(
                            prepared.padX
                        )
                    ) /
                    prepared.scale
                )
            );

        int top =
            static_cast<int>(
                std::round(
                    (
                        centerY -
                        boxHeight / 2.0F -
                        static_cast<float>(
                            prepared.padY
                        )
                    ) /
                    prepared.scale
                )
            );

        int right =
            static_cast<int>(
                std::round(
                    (
                        centerX +
                        boxWidth / 2.0F -
                        static_cast<float>(
                            prepared.padX
                        )
                    ) /
                    prepared.scale
                )
            );

        int bottom =
            static_cast<int>(
                std::round(
                    (
                        centerY +
                        boxHeight / 2.0F -
                        static_cast<float>(
                            prepared.padY
                        )
                    ) /
                    prepared.scale
                )
            );

        /*
         * 박스가 화면 밖으로 나가지 않도록 제한합니다.
         */
        left = std::clamp(
            left,
            0,
            frame.cols - 1
        );

        top = std::clamp(
            top,
            0,
            frame.rows - 1
        );

        right = std::clamp(
            right,
            0,
            frame.cols - 1
        );

        bottom = std::clamp(
            bottom,
            0,
            frame.rows - 1
        );

        /*
         * 너비나 높이가 0 이하가 된 박스는 제거합니다.
         */
        if (
            right <= left ||
            bottom <= top
        ) {
            continue;
        }

        boxes.emplace_back(
            left,
            top,
            right - left,
            bottom - top
        );

        confidences.push_back(
            confidence
        );

        classIds.push_back(
            classId
        );
    }

    /*
     * 같은 추적 범주끼리 묶어서 NMS를 수행합니다.
     *
     * 예:
     * 같은 차량이 car와 truck으로 동시에 검출되면
     * 사륜차 그룹 내부에서 중복 박스를 제거합니다.
     */
    std::map<
        int,
        std::vector<int>
    > indicesByGroup;

    for (
        int index = 0;
        index <
            static_cast<int>(
                classIds.size()
            );
        ++index
    ) {
        const int groupId =
            getNmsGroup(
                classIds[index]
            );

        indicesByGroup[groupId]
            .push_back(index);
    }

    std::vector<Detection> detections;

    for (
        const auto& groupEntry
        : indicesByGroup
    ) {
        const std::vector<int>& globalIndices =
            groupEntry.second;

        std::vector<cv::Rect> groupBoxes;
        std::vector<float> groupConfidences;

        for (
            const int globalIndex
            : globalIndices
        ) {
            groupBoxes.push_back(
                boxes[globalIndex]
            );

            groupConfidences.push_back(
                confidences[globalIndex]
            );
        }

        std::vector<int> keptIndices;

        /*
         * 같은 그룹에서 서로 크게 겹치는 박스 중
         * 신뢰도가 높은 박스를 남깁니다.
         */
        cv::dnn::NMSBoxes(
            groupBoxes,
            groupConfidences,
            confidenceThreshold,
            nmsThreshold,
            keptIndices
        );

        /*
         * NMS 결과의 인덱스는 그룹 내부 기준이므로
         * 전체 검출 결과의 인덱스로 다시 변환합니다.
         */
        for (
            const int localIndex
            : keptIndices
        ) {
            const int globalIndex =
                globalIndices[
                    localIndex
                ];

            detections.push_back({
                classIds[globalIndex],
                confidences[globalIndex],
                boxes[globalIndex]
            });
        }
    }

    return detections;
}

int main(
    int argc,
    char* argv[]
) {
    /*
     * 실행할 때 입력 경로를 전달하지 않으면
     * 기본 영상인 videos/input.mp4를 사용합니다.
     *
     * 예:
     * ./build/adas
     *
     * 다른 영상:
     * ./build/adas videos/test.mp4
     */
    const std::string inputPath =
        argc >= 2
            ? argv[1]
            : "videos/input.mp4";

    const std::string modelPath =
        "models/yolov8n.onnx";

    /*
     * 아직 Step 6을 실행하지 않았으므로
     * 결과 파일도 Step 6으로 저장합니다.
     */
    const std::string outputPath =
        "results/step6_bytetrack_output.avi";

    /*
     * 모델 파일이 없으면 OpenCV 예외가 발생하기 전에
     * 명확한 오류 메시지를 출력하고 종료합니다.
     */
    if (
        !std::filesystem::exists(
            modelPath
        )
    ) {
        std::cerr
            << "[ERROR] YOLO 모델을 찾을 수 없습니다: "
            << modelPath
            << '\n';

        return 1;
    }

    cv::dnn::Net net;

    try {
        /*
         * YOLOv8n ONNX 모델을 OpenCV DNN으로 불러옵니다.
         */
        net =
            cv::dnn::readNetFromONNX(
                modelPath
            );

        /*
         * 현재 단계에서는 OpenCV CPU 추론을 사용합니다.
         */
        net.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );

        net.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );
    } catch (
        const cv::Exception& error
    ) {
        std::cerr
            << "[ERROR] YOLO 모델 로드 실패\n"
            << error.what()
            << '\n';

        return 1;
    }

    std::cout
        << "[SUCCESS] YOLO 모델을 불러왔습니다.\n";

    /*
     * 입력 영상을 엽니다.
     */
    cv::VideoCapture capture(
        inputPath
    );

    if (!capture.isOpened()) {
        std::cerr
            << "[ERROR] 영상을 열 수 없습니다: "
            << inputPath
            << '\n';

        return 1;
    }

    /*
     * 원본 영상의 해상도와 FPS를 읽습니다.
     */
    const int width =
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_WIDTH
            )
        );

    const int height =
        static_cast<int>(
            capture.get(
                cv::CAP_PROP_FRAME_HEIGHT
            )
        );

    double sourceFps =
        capture.get(
            cv::CAP_PROP_FPS
        );

    /*
     * 일부 영상은 FPS 정보를 제대로 반환하지 않을 수 있습니다.
     * 이 경우 기본값으로 30FPS를 사용합니다.
     */
    if (sourceFps <= 0.0) {
        sourceFps = 30.0;
    }

    /*
     * 결과 영상 저장기를 생성합니다.
     *
     * 코덱:
     * MJPG
     *
     * 해상도와 FPS:
     * 원본 영상과 동일
     */
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'
        ),
        sourceFps,
        cv::Size(
            width,
            height
        )
    );

    if (!writer.isOpened()) {
        std::cerr
            << "[ERROR] 결과 영상을 만들 수 없습니다: "
            << outputPath
            << '\n';

        return 1;
    }

    /*
     * 예상 주행 경로를 사다리꼴 ROI로 설정합니다.
     *
     * 고정 픽셀이 아니라 영상 크기의 비율로 지정해
     * 다른 해상도에서도 비슷한 형태를 유지합니다.
     */
    const std::vector<cv::Point> drivingRoi = {
        cv::Point(
            static_cast<int>(
                static_cast<double>(width) *
                0.40
            ),
            static_cast<int>(
                static_cast<double>(height) *
                0.65
            )
        ),

        cv::Point(
            static_cast<int>(
                static_cast<double>(width) *
                0.60
            ),
            static_cast<int>(
                static_cast<double>(height) *
                0.65
            )
        ),

        cv::Point(
            static_cast<int>(
                static_cast<double>(width) *
                0.90
            ),
            static_cast<int>(
                static_cast<double>(height) *
                0.95
            )
        ),

        cv::Point(
            static_cast<int>(
                static_cast<double>(width) *
                0.10
            ),
            static_cast<int>(
                static_cast<double>(height) *
                0.95
            )
        )
    };

    /*
     * YOLO에서 최소 0.10 이상의 검출을 모두 받아옵니다.
     *
     * 추적기에서 다시:
     *
     * 0.25 이상
     * → 고신뢰도 검출
     *
     * 0.10~0.25
     * → 기존 ID 유지용 저신뢰도 검출
     *
     * 로 구분합니다.
     */
    constexpr float detectorThreshold =
        0.10F;

    constexpr float nmsThreshold =
        0.45F;

    /*
     * ByteTrack 방식의 핵심 원리를 적용한 경량 추적기입니다.
     *
     * 0.25:
     * 고신뢰도 검출 기준
     *
     * 0.10:
     * 저신뢰도 최저 기준
     *
     * 3:
     * 고신뢰도 검출이 3회 연결된 후 정식 ID 발급
     *
     * 20:
     * 확정 Track 상태를 최대 20프레임 보관
     */
    MultiObjectTracker tracker(
        0.25F,
        0.10F,
        3,
        20
    );

    cv::Mat frame;

    int processedFrames = 0;

    double totalInferenceMilliseconds =
        0.0;

    const auto totalStart =
        std::chrono::steady_clock::now();

    /*
     * 입력 영상의 모든 프레임을 순서대로 처리합니다.
     */
    while (capture.read(frame)) {
        ++processedFrames;

        /*
         * YOLO 추론 시간만 따로 측정합니다.
         */
        const auto inferenceStart =
            std::chrono::steady_clock::now();

        std::vector<Detection> detections;

        try {
            /*
             * ROI나 글씨가 그려지지 않은 원본 frame으로
             * YOLO 객체 검출을 수행합니다.
             */
            detections =
                detectObjects(
                    net,
                    frame,
                    detectorThreshold,
                    nmsThreshold
                );
        } catch (
            const std::exception& error
        ) {
            std::cerr
                << "[ERROR] 객체 검출 실패: "
                << error.what()
                << '\n';

            return 1;
        }

        const auto inferenceEnd =
            std::chrono::steady_clock::now();

        const double inferenceMilliseconds =
            std::chrono::duration<
                double,
                std::milli
            >(
                inferenceEnd -
                inferenceStart
            ).count();

        totalInferenceMilliseconds +=
            inferenceMilliseconds;

        /*
         * 현재 검출 결과를 이전 프레임의 Track과 연결합니다.
         *
         * 반환값에는 정식 ID가 확정되고,
         * 현재 프레임에서 검출된 객체만 포함됩니다.
         */
        const std::vector<TrackedObject>
            trackedObjects =
                tracker.update(
                    detections
                );

        /*
         * YOLO 추론이 끝난 후에만 ROI를 그립니다.
         *
         * ROI 색상이 객체 검출 입력에 영향을 주지 않도록
         * 처리 순서를 분리합니다.
         */
        cv::Mat overlay =
            frame.clone();

        cv::fillConvexPoly(
            overlay,
            drivingRoi,
            cv::Scalar(
                0,
                255,
                0
            )
        );

        /*
         * ROI를 반투명하게 합성합니다.
         */
        cv::addWeighted(
            overlay,
            0.18,
            frame,
            0.82,
            0.0,
            frame
        );

        cv::polylines(
            frame,
            drivingRoi,
            true,
            cv::Scalar(
                0,
                255,
                0
            ),
            3
        );

        int objectsInsideRoi = 0;

        /*
         * 추적 객체의 바운딩 박스와 ID를 표시합니다.
         */
        for (
            const TrackedObject& trackedObject
            : trackedObjects
        ) {
            const cv::Rect& box =
                trackedObject.box;

            /*
             * 박스 하단 중앙점을 객체가 도로에 닿는
             * 접지점으로 사용합니다.
             */
            const cv::Point groundPoint(
                box.x +
                    box.width / 2,

                box.y +
                    box.height
            );

            /*
             * 접지점이 사다리꼴 ROI 내부에 있는지 확인합니다.
             */
            const bool insideRoi =
                cv::pointPolygonTest(
                    drivingRoi,
                    groundPoint,
                    false
                ) >= 0.0;

            if (insideRoi) {
                ++objectsInsideRoi;
            }

            /*
             * ROI 내부:
             * 빨간색
             *
             * ROI 외부:
             * 주황색
             */
            const cv::Scalar boxColor =
                insideRoi
                    ? cv::Scalar(
                          0,
                          0,
                          255
                      )
                    : cv::Scalar(
                          0,
                          200,
                          255
                      );

            cv::rectangle(
                frame,
                box,
                boxColor,
                3
            );

            /*
             * ROI 판정에 사용한 접지점을 표시합니다.
             */
            cv::circle(
                frame,
                groundPoint,
                6,
                boxColor,
                -1
            );

            /*
             * 예:
             *
             * car ID:3 0.82 IN ROI
             */
            std::string label =
                getClassName(
                    trackedObject.classId
                ) +
                " ID:" +
                std::to_string(
                    trackedObject.trackId
                ) +
                " " +
                cv::format(
                    "%.2f",
                    trackedObject.confidence
                );

            if (insideRoi) {
                label += " IN ROI";
            }

            int baseline = 0;

            const cv::Size labelSize =
                cv::getTextSize(
                    label,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                    &baseline
                );

            /*
             * 박스가 화면 상단에 있더라도
             * 라벨이 영상 밖으로 잘리지 않게 합니다.
             */
            const int labelTop =
                std::max(
                    box.y,
                    labelSize.height + 10
                );

            /*
             * 글씨가 잘 보이도록 라벨 배경을 먼저 그립니다.
             */
            cv::rectangle(
                frame,
                cv::Point(
                    box.x,
                    labelTop -
                        labelSize.height -
                        10
                ),
                cv::Point(
                    box.x +
                        labelSize.width +
                        10,
                    labelTop
                ),
                boxColor,
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(
                    box.x + 5,
                    labelTop - 5
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(
                    0,
                    0,
                    0
                ),
                2
            );
        }

        /*
         * 화면 왼쪽 위에 현재 프레임 번호를 표시합니다.
         */
        cv::putText(
            frame,
            "Frame: " +
                std::to_string(
                    processedFrames
                ),
            cv::Point(
                30,
                45
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                255,
                255,
                255
            ),
            2
        );

        /*
         * 현재 프레임의 YOLO 추론 시간을 표시합니다.
         */
        cv::putText(
            frame,
            "Inference: " +
                cv::format(
                    "%.1f ms",
                    inferenceMilliseconds
                ),
            cv::Point(
                30,
                80
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(
                255,
                255,
                255
            ),
            2
        );

        /*
         * ROI 내부의 추적 객체 수를 표시합니다.
         */
        cv::putText(
            frame,
            "Objects in ROI: " +
                std::to_string(
                    objectsInsideRoi
                ),
            cv::Point(
                30,
                115
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            objectsInsideRoi > 0
                ? cv::Scalar(
                      0,
                      0,
                      255
                  )
                : cv::Scalar(
                      0,
                      255,
                      0
                  ),
            2
        );

        /*
         * 현재는 충돌 위험도가 아니라
         * 객체의 ROI 진입 여부만 경고합니다.
         */
        if (objectsInsideRoi > 0) {
            cv::putText(
                frame,
                "WARNING: OBJECT IN DRIVING PATH",
                cv::Point(
                    static_cast<int>(
                        static_cast<double>(
                            width
                        ) * 0.20
                    ),
                    60
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(
                    0,
                    0,
                    255
                ),
                3
            );
        }

        /*
         * 처리된 프레임을 결과 영상에 저장합니다.
         */
        writer.write(frame);
    }

    const auto totalEnd =
        std::chrono::steady_clock::now();

    /*
     * 영상 전체 처리 시간을 초 단위로 계산합니다.
     */
    const double elapsedSeconds =
        std::chrono::duration<double>(
            totalEnd - totalStart
        ).count();

    /*
     * 실제 프로그램의 전체 처리 속도입니다.
     *
     * 원본 영상 FPS가 아니라
     * 컴퓨터가 초당 몇 프레임을 처리했는지를 뜻합니다.
     */
    const double processingFps =
        elapsedSeconds > 0.0
            ? static_cast<double>(
                  processedFrames
              ) /
                  elapsedSeconds
            : 0.0;

    /*
     * 프레임당 평균 YOLO 추론 시간을 계산합니다.
     */
    const double
        averageInferenceMilliseconds =
            processedFrames > 0
                ? totalInferenceMilliseconds /
                      static_cast<double>(
                          processedFrames
                      )
                : 0.0;

    /*
     * 영상 파일을 정상적으로 마감합니다.
     *
     * writer.release()가 호출돼야 AVI 파일의
     * 인덱스와 길이 정보가 완성됩니다.
     */
    capture.release();
    writer.release();

    std::cout << '\n';

    std::cout
        << "[SUCCESS] YOLO 객체 검출 및 "
        << "Step 6 추적이 완료되었습니다.\n";

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