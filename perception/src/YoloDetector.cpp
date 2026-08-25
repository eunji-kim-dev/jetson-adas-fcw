#include "perception/YoloDetector.hpp"

#include "perception/Classes.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

struct LetterboxResult {
    cv::Mat image;
    float scale;
    int padX;
    int padY;
};

LetterboxResult letterbox(const cv::Mat& frame, int inputSize) {
    const float scale = std::min(static_cast<float>(inputSize) / static_cast<float>(frame.cols), static_cast<float>(inputSize) / static_cast<float>(frame.rows));
    const int resizedWidth = static_cast<int>(std::round(static_cast<float>(frame.cols) * scale));
    const int resizedHeight = static_cast<int>(std::round(static_cast<float>(frame.rows) * scale));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight));

    const int totalPadX = inputSize - resizedWidth;
    const int totalPadY = inputSize - resizedHeight;
    const int padLeft = totalPadX / 2;
    const int padRight = totalPadX - padLeft;
    const int padTop = totalPadY / 2;
    const int padBottom = totalPadY - padTop;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return {padded, scale, padLeft, padTop};
}

/*
 * 전체 프레임 YOLO 결과와 원거리 crop YOLO 결과를 합치면
 * 같은 차량이 두 번 검출될 수 있음
 *
 * 그래서 합친 Detection 전체를
 * car/bus/truck, bicycle/motorcycle, person 그룹으로 다시 묶고
 * IoU 기반 NMS를 한 번 더 수행
 */
std::vector<Detection> applyGroupedNms(const std::vector<Detection>& input, float nmsThreshold) {
    std::map<int, std::vector<int>> indicesByGroup;
    for (int index = 0; index < static_cast<int>(input.size()); ++index) {
        indicesByGroup[getNmsGroup(input[index].classId)].push_back(index);
    }

    std::vector<Detection> result;
    for (const auto& groupEntry : indicesByGroup) {
        const std::vector<int>& globalIndices = groupEntry.second;
        std::vector<cv::Rect> groupBoxes;
        std::vector<float> groupScores;

        for (const int globalIndex : globalIndices) {
            groupBoxes.push_back(input[globalIndex].box);
            groupScores.push_back(input[globalIndex].confidence);
        }

        std::vector<int> keptIndices;
        cv::dnn::NMSBoxes(groupBoxes, groupScores, 0.0F, nmsThreshold, keptIndices);

        for (const int localIndex : keptIndices) {
            result.push_back(input[globalIndices[localIndex]]);
        }
    }
    return result;
}

/*
 * 두 박스 중 더 작은 박스가
 * 다른 박스 안에 얼마나 포함되는지 계산
 *
 * 일반 IoU는 큰 박스와 작은 박스의 크기 차이가 크면
 * 실제로 한 박스가 다른 박스를 많이 감싸고 있어도 낮게 나올 수 있음
 */
float calculateContainment(const cv::Rect& first, const cv::Rect& second) {
    const cv::Rect intersection = first & second;
    const float intersectionArea = static_cast<float>(intersection.area());
    if (intersectionArea <= 0.0F) return 0.0F;

    const float firstArea = static_cast<float>(first.area());
    const float secondArea = static_cast<float>(second.area());
    const float smallerArea = std::max(std::min(firstArea, secondArea), 1.0F);

    return intersectionArea / smallerArea;
}

/*
 * 동영상 17초 ID:29 큰 박스 오검출 오류 수정
 *
 * 프레임 520 실측:
 *   가짜 큰 박스   약 64×18
 *   진짜 작은 박스 약 31×23
 *
 * 기존 NMS         : IoU 약 0.36 < 0.45   → 중복 제거 실패
 * 기존 containment : 약 0.70 < 0.85       → 가로로는 감싸지만 세로가 납작해서 실패
 *
 * 수정:
 *   1. exact class가 아니라 같은 NMS 그룹끼리 비교
 *      car / bus / truck 사이 클래스 흔들림도 같은 차량 후보로 봄
 *   2. containment 0.60 이상이면 부분 포함 후보로 허용
 *   3. 큰 박스 안에 작은 박스 중심이 들어 있고
 *      큰 박스 종횡비가 2.2 이상으로 비정상적으로 납작하면 부분 포함형 중복으로 판단
 *   4. 면적 비율 기준을 1.60 → 1.35로 완화
 *   5. 큰 박스의 형태가 비정상적이면 confidence가 조금 더 높아도 큰 박스를 제거
 */
std::vector<Detection> suppressContainedDuplicates(const std::vector<Detection>& input) {
    constexpr float strictContainment = 0.85F;
    constexpr float looseContainment = 0.60F;
    constexpr float minimumAreaRatio = 1.35F;
    constexpr float largeBoxConfidenceMargin = 0.08F;
    constexpr float abnormalAspectRatio = 2.20F;

    std::vector<bool> removed(input.size(), false);

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (removed[i]) continue;
        for (std::size_t j = i + 1; j < input.size(); ++j) {
            if (removed[j]) continue;

            // 이 추가 필터는 차량 박스에만 적용
            // 보행자/이륜차의 정상적인 겹침에는 영향을 주지 않음
            if (!isVehicleClass(input[i].classId) || !isVehicleClass(input[j].classId)) continue;

            // 같은 클래스만 비교하지 않고
            // car / bus / truck처럼 같은 NMS 그룹이면 비교
            if (getNmsGroup(input[i].classId) != getNmsGroup(input[j].classId)) continue;

            const float containment = calculateContainment(input[i].box, input[j].box);
            if (containment < looseContainment) continue;

            const float firstArea = static_cast<float>(input[i].box.area());
            const float secondArea = static_cast<float>(input[j].box.area());
            const float smallerArea = std::max(std::min(firstArea, secondArea), 1.0F);
            const float largerArea = std::max(firstArea, secondArea);
            const float areaRatio = largerArea / smallerArea;

            if (areaRatio < minimumAreaRatio) continue;

            const std::size_t smallerIndex = firstArea <= secondArea ? i : j;
            const std::size_t largerIndex = firstArea <= secondArea ? j : i;

            const cv::Rect& largerBox = input[largerIndex].box;
            const cv::Rect& smallerBox = input[smallerIndex].box;
            const float largerAspect = static_cast<float>(largerBox.width) / std::max(static_cast<float>(largerBox.height), 1.0F);
            const cv::Point smallerCenter(smallerBox.x + smallerBox.width / 2, smallerBox.y + smallerBox.height / 2);
            const bool centerInside = largerBox.contains(smallerCenter);

            // 0.85 이상이면 강한 포함
            // 0.60~0.85 구간은
            // 작은 박스 중심이 안에 있고 큰 박스 형태가 비정상일 때만 허용
            const bool contained = containment >= strictContainment || (centerInside && largerAspect >= abnormalAspectRatio);
            if (!contained) continue;

            // 큰 박스 형태가 비정상적이면
            // confidence가 조금 높더라도 큰 박스를 우선 제거
            if (largerAspect >= abnormalAspectRatio || input[largerIndex].confidence <= input[smallerIndex].confidence + largeBoxConfidenceMargin) {
                removed[largerIndex] = true;
            } else {
                removed[smallerIndex] = true;
            }
            if (removed[i]) break;
        }
    }

    std::vector<Detection> result;
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (!removed[index]) result.push_back(input[index]);
    }
    return result;
}

} // namespace

YoloDetector::YoloDetector(const std::string& modelPath, float confidenceThreshold, float nmsThreshold)
    : confidenceThreshold_(confidenceThreshold), nmsThreshold_(nmsThreshold) {
    net_ = cv::dnn::readNetFromONNX(modelPath);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame, DetectionTiming* timing) {
    // 기본 YOLO 검출
    // 전체 블랙박스 프레임에서 먼저 객체를 검출
    const auto fullYoloStart = std::chrono::steady_clock::now();
    std::vector<Detection> detections = runInference(frame);
    const auto fullYoloEnd = std::chrono::steady_clock::now();

    // 전체 프레임에서는 너무 작아진 원거리 차량을
    // 중앙 도로 crop 영역에서 한 번 더 확대 추론
    const auto farYoloStart = std::chrono::steady_clock::now();
    const std::vector<Detection> farDetections = detectFarRoadObjects(frame);
    const auto farYoloEnd = std::chrono::steady_clock::now();

    const auto postprocessStart = std::chrono::steady_clock::now();
    // 전체 프레임 검출 + 원거리 crop 검출을 합침
    detections.insert(detections.end(), farDetections.begin(), farDetections.end());
    // 두 추론에서 같은 차량이 각각 검출될 수 있으므로
    // 합쳐진 결과에 클래스 그룹별 NMS를 다시 적용
    detections = applyGroupedNms(detections, nmsThreshold_);
    // IoU NMS로 제거되지 않는 부분 포함형/납작한 큰 박스를
    // containment + 중심 포함 + 종횡비 조건으로 한 번 더 제거
    detections = suppressContainedDuplicates(detections);

    // 높이 4~5px 수준의 극소 박스가 Track으로 확정되는 것을 방지
    const int minimumDetectionHeight = std::max(8, static_cast<int>(std::round(frame.rows * 0.014)));
    detections.erase(std::remove_if(detections.begin(), detections.end(), [&](const Detection& detection) {
        return detection.box.height < minimumDetectionHeight;
    }), detections.end());
    const auto postprocessEnd = std::chrono::steady_clock::now();

    if (timing != nullptr) {
        timing->fullYoloMilliseconds = std::chrono::duration<double, std::milli>(fullYoloEnd - fullYoloStart).count();
        timing->farYoloMilliseconds = std::chrono::duration<double, std::milli>(farYoloEnd - farYoloStart).count();
        timing->postprocessMilliseconds = std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count();
    }

    return detections;
}
std::vector<Detection> YoloDetector::runInference(const cv::Mat& frame) {
    constexpr int inputSize = 640;
    const LetterboxResult prepared = letterbox(frame, inputSize);

    cv::Mat blob = cv::dnn::blobFromImage(prepared.image, 1.0 / 255.0, cv::Size(inputSize, inputSize), cv::Scalar(), true, false);
    net_.setInput(blob);
    cv::Mat output = net_.forward();

    if (output.dims != 3) throw std::runtime_error("지원하지 않는 YOLO 출력 차원");

    const int firstDimension = output.size[1];
    const int secondDimension = output.size[2];
    cv::Mat predictions(firstDimension, secondDimension, CV_32F, output.ptr<float>());
    
    if (firstDimension < secondDimension) predictions = predictions.t();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int row = 0; row < predictions.rows; ++row) {
        const float* data = predictions.ptr<float>(row);
        const float centerX = data[0], centerY = data[1], boxWidth = data[2], boxHeight = data[3];

        cv::Mat classScores(1, predictions.cols - 4, CV_32F, const_cast<float*>(data + 4));
        cv::Point bestClassPoint;
        double bestClassScore = 0.0;
        cv::minMaxLoc(classScores, nullptr, &bestClassScore, nullptr, &bestClassPoint);

        const int rawClassId = bestClassPoint.x;
        const float rawConfidence = static_cast<float>(bestClassScore);

        if (rawConfidence < confidenceThreshold_) continue;

        int left = static_cast<int>(std::round((centerX - boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int top = static_cast<int>(std::round((centerY - boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));
        int right = static_cast<int>(std::round((centerX + boxWidth / 2.0F - static_cast<float>(prepared.padX)) / prepared.scale));
        int bottom = static_cast<int>(std::round((centerY + boxHeight / 2.0F - static_cast<float>(prepared.padY)) / prepared.scale));

        left = std::clamp(left, 0, frame.cols - 1);
        top = std::clamp(top, 0, frame.rows - 1);
        right = std::clamp(right, 0, frame.cols - 1);
        bottom = std::clamp(bottom, 0, frame.rows - 1);

        if (right <= left || bottom <= top) continue;

        const int restoredWidth = right - left;
        const int restoredHeight = bottom - top;

        if (!isTargetClass(rawClassId)) continue;

        boxes.emplace_back(left, top, restoredWidth, restoredHeight);
        confidences.push_back(rawConfidence);
        classIds.push_back(rawClassId);
    }

    std::map<int, std::vector<int>> indicesByGroup;
    for (int index = 0; index < static_cast<int>(classIds.size()); ++index) {
        indicesByGroup[getNmsGroup(classIds[index])].push_back(index);
    }

    std::vector<Detection> detections;
    for (const auto& groupEntry : indicesByGroup) {
        const std::vector<int>& globalIndices = groupEntry.second;
        std::vector<cv::Rect> groupBoxes;
        std::vector<float> groupConfidences;

        for (const int globalIndex : globalIndices) {
            groupBoxes.push_back(boxes[globalIndex]);
            groupConfidences.push_back(confidences[globalIndex]);
        }

        std::vector<int> keptIndices;
        cv::dnn::NMSBoxes(groupBoxes, groupConfidences, confidenceThreshold_, nmsThreshold_, keptIndices);

        for (const int localIndex : keptIndices) {
            const int globalIndex = globalIndices[localIndex];
            detections.push_back({classIds[globalIndex], confidences[globalIndex], boxes[globalIndex]});
        }
    }

    return detections;
}

/*
 * 동영상 13초 원거리 차량 미검출 오류 수정
 *
 * 전체 1280×720 프레임을 YOLO의 640×640 입력으로 축소하면
 * 멀리 있는 작은 차량은 수 픽셀 수준으로 줄어들어 검출이 끊길 수 있음
 *
 * 그래서 화면 중앙의 원거리 도로 영역만 crop한 뒤
 * 같은 YOLOv8n 모델로 한 번 더 추론
 *
 * crop에서 나온 좌표는 다시 원본 프레임 좌표로 복원
 */
std::vector<Detection> YoloDetector::detectFarRoadObjects(const cv::Mat& frame) {
    const int cropX = static_cast<int>(std::round(frame.cols * 0.25F));
    const int cropY = static_cast<int>(std::round(frame.rows * 0.38F));
    const int cropWidth = static_cast<int>(std::round(frame.cols * 0.50F));
    const int cropHeight = static_cast<int>(std::round(frame.rows * 0.36F));

    const int safeX = std::clamp(cropX, 0, frame.cols - 1);
    const int safeY = std::clamp(cropY, 0, frame.rows - 1);
    const int safeWidth = std::min(cropWidth, frame.cols - safeX);
    const int safeHeight = std::min(cropHeight, frame.rows - safeY);

    if (safeWidth <= 1 || safeHeight <= 1) return {};

    const cv::Rect cropRect(safeX, safeY, safeWidth, safeHeight);
    const cv::Mat crop = frame(cropRect).clone();

    std::vector<Detection> cropDetections = runInference(crop);
    std::vector<Detection> result;

    for (Detection detection : cropDetections) {
        // 동영상 38초 ID:60 오검출 수정
        // crop은 원거리 차량 보조 검출용이라 너무 낮은 confidence 후보는 제외
        if (detection.confidence < 0.20F) continue;
        
        // 원거리 보조 추론에서는 차량 계열만 사용
        // person 등까지 중복 추론해서 오검출이 늘어나는 것을 줄임
        if (!isVehicleClass(detection.classId)) continue;

        // crop 내부 좌표를 원본 영상 좌표로 복원
        detection.box.x += cropRect.x;
        detection.box.y += cropRect.y;
        result.push_back(detection);
    }

    return result;
}

