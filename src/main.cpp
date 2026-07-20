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

// letterbox 결과물. 640x640으로 맞춘 이미지랑
// 나중에 좌표 복원할 때 필요한 배율/여백을 같이 들고 다님
struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

// 검출된 객체 하나
struct Detection {
    int classId;
    float confidence;
    cv::Rect box;
};

// COCO 80개 클래스 중 ADAS에서 볼 것만 통과시킴
// 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
bool isTargetClass(int classId) {
    return classId == 0 ||
           classId == 1 ||
           classId == 2 ||
           classId == 3 ||
           classId == 5 ||
           classId == 7;
}

// 클래스 번호 -> 이름
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

// 비율 유지한 채로 640x640에 맞추고, 남는 공간은 회색(114)으로 패딩
// 그냥 resize로 찌그러뜨리면 검출 정확도가 떨어져서 이 방식 씀
LetterboxResult letterbox(
    const cv::Mat& frame,
    int inputSize
) {
    const float scale = std::min(
        static_cast<float>(inputSize) / frame.cols,
        static_cast<float>(inputSize) / frame.rows
    );

    const int resizedWidth =
        static_cast<int>(std::round(frame.cols * scale));

    const int resizedHeight =
        static_cast<int>(std::round(frame.rows * scale));

    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(resizedWidth, resizedHeight)
    );

    // 여백은 상하/좌우 반씩 나눠서 가운데 정렬
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

// YOLO 한 번 돌려서 검출 결과 리스트로 반환
std::vector<Detection> detectObjects(
    cv::dnn::Net& net,
    const cv::Mat& frame,
    float confidenceThreshold,
    float nmsThreshold
) {
    constexpr int inputSize = 640;

    const LetterboxResult prepared =
        letterbox(frame, inputSize);

    // 이미지 -> 입력 텐서 변환
    // 1/255: 픽셀값 0~1로 정규화, swapRB: BGR -> RGB (YOLO는 RGB 기준)
    cv::Mat blob = cv::dnn::blobFromImage(
        prepared.image,
        1.0 / 255.0,
        cv::Size(inputSize, inputSize),
        cv::Scalar(),
        true,
        false
    );

    net.setInput(blob);

    // 실제 추론이 도는 지점. 시간 대부분 여기서 씀
    cv::Mat output = net.forward();

    if (output.dims != 3) {
        throw std::runtime_error(
            "지원하지 않는 YOLO 출력 차원입니다."
        );
    }

    // yolov8 onnx 출력이 [1, 84, 8400]일 수도 [1, 8400, 84]일 수도 있음
    // 후보 8400개가 행, (좌표4 + 클래스80)이 열이 되게 방향 통일
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
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int row = 0; row < predictions.rows; ++row) {
        const float* data =
            predictions.ptr<float>(row);

        // 앞 4개: 박스 (중심x, 중심y, 폭, 높이)
        const float centerX = data[0];
        const float centerY = data[1];
        const float boxWidth = data[2];
        const float boxHeight = data[3];

        // 4번부터 끝까지가 클래스별 점수. 제일 높은 것만 씀
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

        const int classId = bestClassPoint.x;
        const float confidence =
            static_cast<float>(bestClassScore);

        if (confidence < confidenceThreshold) {
            continue;
        }

        if (!isTargetClass(classId)) {
            continue;
        }

        // 640 기준 좌표 -> 원본 좌표 복원
        // letterbox 여백 먼저 빼고, resize 배율로 나누는 순서
        int left = static_cast<int>(
            std::round(
                (
                    centerX -
                    boxWidth / 2.0F -
                    prepared.padX
                ) / prepared.scale
            )
        );

        int top = static_cast<int>(
            std::round(
                (
                    centerY -
                    boxHeight / 2.0F -
                    prepared.padY
                ) / prepared.scale
            )
        );

        int right = static_cast<int>(
            std::round(
                (
                    centerX +
                    boxWidth / 2.0F -
                    prepared.padX
                ) / prepared.scale
            )
        );

        int bottom = static_cast<int>(
            std::round(
                (
                    centerY +
                    boxHeight / 2.0F -
                    prepared.padY
                ) / prepared.scale
            )
        );

        // 화면 밖으로 나간 좌표 잘라내기
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

        // 클램프 후 뒤집힌 박스는 버림
        if (right <= left || bottom <= top) {
            continue;
        }

        boxes.emplace_back(
            left,
            top,
            right - left,
            bottom - top
        );

        confidences.push_back(confidence);
        classIds.push_back(classId);
    }

    // 같은 물체에 박스가 여러 개 겹쳐 나오는 걸 NMS로 정리
    // 전체를 한 번에 돌리면 다른 클래스끼리 지워버릴 수 있어서 클래스별로 따로 돌림
    std::map<int, std::vector<int>> indicesByClass;

    for (int index = 0;
         index < static_cast<int>(classIds.size());
         ++index) {
        indicesByClass[classIds[index]].push_back(index);
    }

    std::vector<Detection> detections;

    for (const auto& [classId, globalIndices]
         : indicesByClass) {
        std::vector<cv::Rect> classBoxes;
        std::vector<float> classConfidences;

        for (const int globalIndex : globalIndices) {
            classBoxes.push_back(boxes[globalIndex]);
            classConfidences.push_back(
                confidences[globalIndex]
            );
        }

        std::vector<int> keptIndices;

        cv::dnn::NMSBoxes(
            classBoxes,
            classConfidences,
            confidenceThreshold,
            nmsThreshold,
            keptIndices
        );

        // NMS가 준 인덱스는 클래스 내부 기준이라 전체 인덱스로 되돌려야 함
        for (const int localIndex : keptIndices) {
            const int globalIndex =
                globalIndices[localIndex];

            detections.push_back({
                classId,
                confidences[globalIndex],
                boxes[globalIndex]
            });
        }
    }

    return detections;
}

int main(int argc, char* argv[]) {
    // 실행할 때 경로 안 넘기면 videos/input.mp4 사용
    const std::string inputPath =
        argc >= 2
            ? argv[1]
            : "videos/input.mp4";

    const std::string modelPath =
        "models/yolov8n.onnx";

    const std::string outputPath =
        "results/step4_yolo_output.avi";

    // 모델 파일부터 확인. 없으면 readNet에서 예외 나기 전에 여기서 끝냄
    if (!std::filesystem::exists(modelPath)) {
        std::cerr
            << "[ERROR] YOLO 모델을 찾을 수 없습니다: "
            << modelPath << '\n';

        return 1;
    }

    cv::dnn::Net net;

    try {
        // onnx 로드. 이번 단계는 CPU 추론으로 감
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
            << error.what() << '\n';

        return 1;
    }

    std::cout
        << "[SUCCESS] YOLO 모델을 불러왔습니다.\n";

    cv::VideoCapture capture(inputPath);

    if (!capture.isOpened()) {
        std::cerr
            << "[ERROR] 영상을 열 수 없습니다: "
            << inputPath << '\n';

        return 1;
    }

    // 원본 영상 정보 (해상도, fps)
    const int width = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_WIDTH)
    );

    const int height = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    double sourceFps =
        capture.get(cv::CAP_PROP_FPS);

    // 영상에 따라 fps가 0으로 나오는 경우가 있어서 그때는 30으로 가정
    if (sourceFps <= 0.0) {
        sourceFps = 30.0;
    }

    // 결과 저장용 writer. 코덱은 MJPG, 나머지는 원본이랑 동일하게
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc(
            'M',
            'J',
            'P',
            'G'
        ),
        sourceFps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr
            << "[ERROR] 결과 영상을 만들 수 없습니다.\n";

        return 1;
    }

    // step3 ROI에서 윗변만 65%로 내린 버전
    // 55%는 지평선 근처까지 올라가서 먼 차까지 다 잡혀버림
    const std::vector<cv::Point> drivingRoi = {
        cv::Point(
            static_cast<int>(width * 0.40),
            static_cast<int>(height * 0.65)
        ),
        cv::Point(
            static_cast<int>(width * 0.60),
            static_cast<int>(height * 0.65)
        ),
        cv::Point(
            static_cast<int>(width * 0.90),
            static_cast<int>(height * 0.95)
        ),
        cv::Point(
            static_cast<int>(width * 0.10),
            static_cast<int>(height * 0.95)
        )
    };

    constexpr float confidenceThreshold = 0.25F;
    constexpr float nmsThreshold = 0.45F;

    cv::Mat frame;

    int processedFrames = 0;
    double totalInferenceMilliseconds = 0.0;

    const auto totalStart =
        std::chrono::steady_clock::now();

    while (capture.read(frame)) {
        ++processedFrames;

        // 추론 시간은 전체 시간이랑 별도로 측정
        const auto inferenceStart =
            std::chrono::steady_clock::now();

        std::vector<Detection> detections;

        try {
            // 아직 아무것도 그리지 않은 원본 frame으로 검출
            detections = detectObjects(
                net,
                frame,
                confidenceThreshold,
                nmsThreshold
            );
        } catch (const std::exception& error) {
            std::cerr
                << "[ERROR] 객체 검출 실패: "
                << error.what() << '\n';

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

        // ROI 반투명 표시 (step3와 동일, 박스가 주인공이라 투명도만 낮춤)
        cv::Mat overlay = frame.clone();

        cv::fillConvexPoly(
            overlay,
            drivingRoi,
            cv::Scalar(0, 255, 0)
        );

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
            cv::Scalar(0, 255, 0),
            3
        );

        int objectsInsideRoi = 0;

        for (const Detection& detection : detections) {
            // ROI 판정 기준점은 박스 중심이 아니라 하단 중앙
            // 물체가 실제로 바닥에 닿는 지점이라 도로 위에 있는지 보기엔 이게 맞음
            const cv::Point groundPoint(
                detection.box.x +
                    detection.box.width / 2,
                detection.box.y +
                    detection.box.height
            );

            const bool insideRoi =
                cv::pointPolygonTest(
                    drivingRoi,
                    groundPoint,
                    false
                ) >= 0.0;

            if (insideRoi) {
                ++objectsInsideRoi;
            }

            // ROI 안이면 빨강, 밖이면 주황 (BGR 순서 주의)
            const cv::Scalar boxColor =
                insideRoi
                    ? cv::Scalar(0, 0, 255)
                    : cv::Scalar(0, 200, 255);

            cv::rectangle(
                frame,
                detection.box,
                boxColor,
                3
            );

            // ROI 판정에 쓴 하단 중앙점도 같이 찍어줌
            cv::circle(
                frame,
                groundPoint,
                6,
                boxColor,
                -1
            );

            std::string label =
                getClassName(detection.classId) +
                " " +
                cv::format(
                    "%.2f",
                    detection.confidence
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

            // 박스가 화면 맨 위에 붙어 있으면 라벨이 잘려서 최소 높이 보장
            const int labelTop = std::max(
                detection.box.y,
                labelSize.height + 10
            );

            // 라벨 배경 깔고 그 위에 검은 글씨
            cv::rectangle(
                frame,
                cv::Point(
                    detection.box.x,
                    labelTop -
                        labelSize.height -
                        10
                ),
                cv::Point(
                    detection.box.x +
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
                    detection.box.x + 5,
                    labelTop - 5
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 0, 0),
                2
            );
        }

        // 좌상단 상태 표시 (프레임 번호, 추론 시간, ROI 내 객체 수)
        cv::putText(
            frame,
            "Frame: " +
                std::to_string(processedFrames),
            cv::Point(30, 45),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            2
        );

        cv::putText(
            frame,
            "Inference: " +
                cv::format(
                    "%.1f ms",
                    inferenceMilliseconds
                ),
            cv::Point(30, 80),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(255, 255, 255),
            2
        );

        // ROI에 뭐라도 있으면 빨간 글씨로 바뀜
        cv::putText(
            frame,
            "Objects in ROI: " +
                std::to_string(objectsInsideRoi),
            cv::Point(30, 115),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            objectsInsideRoi > 0
                ? cv::Scalar(0, 0, 255)
                : cv::Scalar(0, 255, 0),
            2
        );

        // 주행 경로에 객체 있으면 상단 중앙에 경고 문구
        if (objectsInsideRoi > 0) {
            cv::putText(
                frame,
                "WARNING: OBJECT IN DRIVING PATH",
                cv::Point(
                    static_cast<int>(width * 0.20),
                    60
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 0, 255),
                3
            );
        }

        writer.write(frame);
    }

    const auto totalEnd =
        std::chrono::steady_clock::now();

    // 전체 걸린 시간(초)
    const double elapsedSeconds =
        std::chrono::duration<double>(
            totalEnd - totalStart
        ).count();

    const double processingFps =
        elapsedSeconds > 0.0
            ? processedFrames / elapsedSeconds
            : 0.0;

    // 프레임당 평균 추론 시간
    const double averageInferenceMilliseconds =
        processedFrames > 0
            ? totalInferenceMilliseconds /
                processedFrames
            : 0.0;

    capture.release();
    writer.release();

    // 처리 결과 요약 출력
    std::cout << '\n';
    std::cout
        << "[SUCCESS] YOLO 객체 검출이 완료되었습니다.\n";

    std::cout
        << "처리 프레임 수: "
        << processedFrames << '\n';

    std::cout
        << "평균 추론 시간: "
        << std::fixed << std::setprecision(2)
        << averageInferenceMilliseconds
        << " ms\n";

    std::cout
        << "전체 처리 속도: "
        << std::fixed << std::setprecision(2)
        << processingFps
        << " FPS\n";

    std::cout
        << "결과 파일: "
        << outputPath << '\n';

    return 0;
}