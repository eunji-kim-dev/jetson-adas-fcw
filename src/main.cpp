int main(int argc, char* argv[]) {
    const std::string inputPath = argc >= 2 ? argv[1] : "videos/input.mp4";
    const std::string modelPath = "models/yolov8n.onnx";
    const std::string outputPath = "results/step7_scene_change_fixed_output.avi";

    std::filesystem::create_directories("results");

    if (!std::filesystem::exists(modelPath)) {
        std::cerr << "[ERROR] YOLO 모델을 찾을 수 없습니다: " << modelPath << '\n';
        return 1;
    }

    KoreanTextRenderer koreanText;

    if (!koreanText.initialize()) {
        std::cerr << "[ERROR] 한글 글꼴을 찾거나 불러오지 못했습니다.\n"
                  << "다음 명령으로 필요한 패키지를 설치하세요:\n"
                  << "sudo apt install -y libopencv-contrib-dev fonts-nanum\n";
        return 1;
    }

    std::cout << "[SUCCESS] 한글 글꼴: " << koreanText.fontPath() << '\n';

    cv::dnn::Net net;

    try {
        net = cv::dnn::readNetFromONNX(modelPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception& error) {
        std::cerr << "[ERROR] YOLO 모델 로드 실패\n" << error.what() << '\n';
        return 1;
    }

    cv::VideoCapture capture(inputPath);

    if (!capture.isOpened()) {
        std::cerr << "[ERROR] 영상을 열 수 없습니다: " << inputPath << '\n';
        return 1;
    }

    const int width  = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    double sourceFps = capture.get(cv::CAP_PROP_FPS);

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
        std::cerr << "[ERROR] 결과 영상을 만들 수 없습니다: " << outputPath << '\n';
        return 1;
    }

    // 전체 도로 영역
    // 위험 판단은 이 영역 안의 모든 차량이 아니라
    // 아래 egoLaneRoi 안에서 선택된 선행 차량 한 대를 기준으로 함
    const std::vector<cv::Point> roadRoi = {
        cv::Point(static_cast<int>(width * 0.37F), static_cast<int>(height * 0.64F)),
        cv::Point(static_cast<int>(width * 0.64F), static_cast<int>(height * 0.64F)),
        cv::Point(static_cast<int>(width * 0.96F), static_cast<int>(height * 0.96F)),
        cv::Point(static_cast<int>(width * 0.04F), static_cast<int>(height * 0.96F))
    };

    // 내 차량이 주행하는 차선 영역
    // 차량 박스의 하단 중앙 접지점을 기준으로 ego lane 내부인지 판단함
    const std::vector<cv::Point> egoLaneRoi = {
        cv::Point(static_cast<int>(width * 0.40F), static_cast<int>(height * 0.65F)),
        cv::Point(static_cast<int>(width * 0.48F), static_cast<int>(height * 0.65F)),
        cv::Point(static_cast<int>(width * 0.68F), static_cast<int>(height * 0.95F)),
        cv::Point(static_cast<int>(width * 0.10F), static_cast<int>(height * 0.95F))
    };

    constexpr float detectorThreshold = 0.10F;
    constexpr float nmsThreshold      = 0.45F;

    MultiObjectTracker tracker(0.25F, 0.10F, 3, 20);
    RiskAnalyzer riskAnalyzer(sourceFps, 15, 60);

    // ============================================================
    // 13초 / 17초 검출 오류 보완용 함수
    // ============================================================

    // 동영상 13초 차량 미검출 오류 수정
    // 전체 프레임을 YOLO 640x640 입력으로 줄이면
    // 멀리 있는 차량의 박스가 너무 작아져 검출이 끊기는 문제가 있음
    // 화면 중앙의 원거리 도로 부분만 crop해서 한 번 더 YOLO 추론
    // crop에서는 같은 차량이 전체 프레임 추론보다 크게 보이게 됨
    // crop 좌표로 나온 Detection은 다시 원본 프레임 좌표로 복원함
    const auto detectFarRoadObjects = [&](const cv::Mat& sourceFrame) -> std::vector<Detection> {
        const int cropX      = static_cast<int>(std::round(sourceFrame.cols * 0.25F));
        const int cropY      = static_cast<int>(std::round(sourceFrame.rows * 0.38F));
        const int cropWidth  = static_cast<int>(std::round(sourceFrame.cols * 0.50F));
        const int cropHeight = static_cast<int>(std::round(sourceFrame.rows * 0.36F));

        const int safeX = std::clamp(cropX, 0, sourceFrame.cols - 1);
        const int safeY = std::clamp(cropY, 0, sourceFrame.rows - 1);

        const int safeWidth  = std::min(cropWidth,  sourceFrame.cols - safeX);
        const int safeHeight = std::min(cropHeight, sourceFrame.rows - safeY);

        if (safeWidth <= 1 || safeHeight <= 1) {
            return {};
        }

        const cv::Rect cropRect(safeX, safeY, safeWidth, safeHeight);
        const cv::Mat crop = sourceFrame(cropRect).clone();

        std::vector<Detection> cropDetections =
            detectObjects(net, crop, detectorThreshold, nmsThreshold);

        std::vector<Detection> result;

        for (Detection detection : cropDetections) {
            // 원거리 보조 추론은 차량 계열만 사용
            // person 같은 객체까지 두 번 검출해서
            // 불필요한 오검출이 늘어나는 것을 줄임
            if (!isVehicleClass(detection.classId)) {
                continue;
            }

            // crop 내부 좌표를 원본 블랙박스 영상 좌표로 복원
            detection.box.x += cropRect.x;
            detection.box.y += cropRect.y;

            result.push_back(detection);
        }

        return result;
    };

    // 전체 프레임 YOLO 결과와 원거리 crop YOLO 결과를 합치면
    // 같은 차량이 두 번 검출될 수 있기 때문에
    // 합친 Detection 전체에 클래스 그룹별 NMS를 한 번 더 수행함
    const auto applyGroupedNms =
        [&](const std::vector<Detection>& input, float threshold) -> std::vector<Detection> {
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
            cv::dnn::NMSBoxes(groupBoxes, groupScores, 0.0F, threshold, keptIndices);

            for (const int localIndex : keptIndices) {
                result.push_back(input[globalIndices[localIndex]]);
            }
        }

        return result;
    };

    // 동영상 17초 ID:29처럼
    // 큰 가짜 박스가 실제 차량의 작은 박스를 감싸는 경우를 확인하기 위한 값
    // 일반 IoU는 큰 박스와 작은 박스의 면적 차이가 크면
    // 작은 박스가 거의 완전히 포함되어 있어도 값이 낮게 나올 수 있음
    // 그래서 교집합 / 더 작은 박스 면적으로 포함 정도를 추가 계산함
    const auto calculateContainment = [](const cv::Rect& first, const cv::Rect& second) -> float {
        const cv::Rect intersection = first & second;
        const float intersectionArea = static_cast<float>(intersection.area());

        if (intersectionArea <= 0.0F) {
            return 0.0F;
        }

        const float firstArea  = static_cast<float>(first.area());
        const float secondArea = static_cast<float>(second.area());
        const float smallerArea = std::max(std::min(firstArea, secondArea), 1.0F);

        return intersectionArea / smallerArea;
    };

    // 동영상 17초 ID:29 중복 오검출 오류 수정
    // 일반 NMS는 IoU 0.45를 기준으로 중복 박스를 제거하지만
    // 큰 가짜 박스가 작은 실제 차량 박스를 감싸면 IoU가 낮아 둘 다 남을 수 있음
    // 작은 박스의 85% 이상이 큰 박스 안에 들어가고
    // 두 박스의 면적 차이가 1.6배 이상이면 포함형 중복 후보로 판단
    // 큰 박스 신뢰도가 작은 박스보다 확실하게 높지 않으면
    // 실제 차량보다 지나치게 크게 잡힌 큰 박스를 제거함
    const auto suppressContainedDuplicates =
        [&](const std::vector<Detection>& input) -> std::vector<Detection> {
        constexpr float containmentThreshold   = 0.85F;
        constexpr float minimumAreaRatio       = 1.60F;
        constexpr float largeBoxConfidenceMargin = 0.08F;

        std::vector<bool> removed(input.size(), false);

        for (std::size_t i = 0; i < input.size(); ++i) {
            if (removed[i]) {
                continue;
            }

            for (std::size_t j = i + 1; j < input.size(); ++j) {
                if (removed[j]) {
                    continue;
                }

                // 다른 클래스끼리는 강제로 포함형 중복 제거를 하지 않음
                if (input[i].classId != input[j].classId) {
                    continue;
                }

                const float containment = calculateContainment(input[i].box, input[j].box);

                if (containment < containmentThreshold) {
                    continue;
                }

                const float firstArea  = static_cast<float>(input[i].box.area());
                const float secondArea = static_cast<float>(input[j].box.area());

                const float smallerArea = std::max(std::min(firstArea, secondArea), 1.0F);
                const float largerArea  = std::max(firstArea, secondArea);
                const float areaRatio   = largerArea / smallerArea;

                if (areaRatio < minimumAreaRatio) {
                    continue;
                }

                const std::size_t smallerIndex = firstArea <= secondArea ? i : j;
                const std::size_t largerIndex  = firstArea <= secondArea ? j : i;

                const Detection& smallerDetection = input[smallerIndex];
                const Detection& largerDetection  = input[largerIndex];

                // 큰 박스의 confidence가
                // 작은 박스보다 0.08 이상 확실히 높은 경우가 아니라면
                // 실제 객체를 과도하게 크게 감싼 박스로 보고 큰 박스를 제거
                if (largerDetection.confidence <=
                    smallerDetection.confidence + largeBoxConfidenceMargin) {
                    removed[largerIndex] = true;
                } else {
                    // 큰 박스의 신뢰도가 확실히 높으면
                    // 작은 부분 박스를 중복으로 보고 작은 쪽 제거
                    removed[smallerIndex] = true;
                }

                if (removed[i]) {
                    break;
                }
            }
        }

        std::vector<Detection> result;

        for (std::size_t index = 0; index < input.size(); ++index) {
            if (!removed[index]) {
                result.push_back(input[index]);
            }
        }

        return result;
    };

    int activeLeadId = -1;

    // 경고 문구가 한두 프레임 만에 사라져 깜빡이지 않도록
    // 확정된 경고는 약 0.25초 동안 유지
    const int warningHoldFrames =
        std::max(1, static_cast<int>(std::round(sourceFps * 0.25)));

    // 멀리 있는 작은 차량의 박스 흔들림 때문에
    // 잘못된 TTC-P 경고가 발생하는 것을 막기 위한 최종 경고 조건
    const int minimumLeadBoxHeightForWarning =
        std::max(36, static_cast<int>(std::round(height * 0.05)));

    const int minimumLeadGroundYForWarning =
        static_cast<int>(std::round(height * 0.60));

    constexpr int minimumRiskSamplesForWarning = 10;
    constexpr float minimumHeightGrowthRatioForWarning = 1.08F;
    constexpr float minimumGroundSpeedForWarning = 2.0F;

    int cautionHoldRemaining = 0;
    int dangerHoldRemaining  = 0;

    // RiskAnalyzer 내부 안정화 이후에도
    // 상단 경고 문구는 같은 상태가 일정 프레임 연속 나와야 표시
    constexpr int cautionConfirmationFrames = 8;
    constexpr int dangerConfirmationFrames  = 3;

    int cautionCandidateFrames = 0;
    int dangerCandidateFrames  = 0;
    int warningCandidateLeadId = -1;

    // 장면 전환 직후 이전 장면의 TTC-P 이력이 이어지는 것을 막기 위해
    // 약 0.5초 동안 선행차 위험 분석을 중지
    const int sceneWarmupFrames =
        std::max(1, static_cast<int>(std::round(sourceFps * 0.5)));

    int sceneWarmupRemaining = 0;

    SceneChangeDetector sceneChangeDetector;

    cv::Mat frame;
    int processedFrames = 0;
    double totalInferenceMilliseconds = 0.0;

    const auto totalStart = std::chrono::steady_clock::now();

    while (capture.read(frame)) {
        ++processedFrames;

        // ROI나 박스를 그리기 전
        // 깨끗한 원본 프레임으로 장면 전환 여부를 먼저 확인
        float sceneDifference = 0.0F;
        float sceneHistogramCorrelation = 1.0F;

        const bool sceneChanged =
            sceneChangeDetector.update(frame, &sceneDifference, &sceneHistogramCorrelation);

        if (sceneChanged) {
            // 장면이 바뀌면
            // 이전 장면에서 유지되던 경고와 선행차 정보를 초기화
            cautionHoldRemaining   = 0;
            dangerHoldRemaining    = 0;
            cautionCandidateFrames = 0;
            dangerCandidateFrames  = 0;
            warningCandidateLeadId = -1;
            activeLeadId           = -1;
            sceneWarmupRemaining   = sceneWarmupFrames;

            std::cout << "[SCENE CHANGE] frame=" << processedFrames
                      << " diff=" << std::fixed << std::setprecision(2) << sceneDifference
                      << " histogram=" << sceneHistogramCorrelation << '\n';
        }

        const bool riskAnalysisEnabled = sceneWarmupRemaining <= 0;

        const auto inferenceStart = std::chrono::steady_clock::now();

        std::vector<Detection> detections;

        try {
            // 기본 YOLO 검출
            // 전체 블랙박스 프레임을 대상으로 먼저 객체를 검출
            detections = detectObjects(net, frame, detectorThreshold, nmsThreshold);

            // 동영상 13초 차량 미검출 오류 수정
            // 전체 프레임에서는 너무 작아진 원거리 차량을
            // 중앙 도로 crop 영역에서 한 번 더 확대 추론
            const std::vector<Detection> farDetections = detectFarRoadObjects(frame);

            // 전체 프레임 검출 결과와 원거리 crop 검출 결과를 하나로 합침
            detections.insert(detections.end(), farDetections.begin(), farDetections.end());

            // 전체 추론과 crop 추론에서
            // 같은 차량이 각각 한 번씩 검출될 수 있으므로
            // 합쳐진 전체 결과에 NMS를 다시 적용
            detections = applyGroupedNms(detections, nmsThreshold);

            // 동영상 17초 ID:29 오류 수정
            // 일반 IoU NMS에서 살아남은
            // 큰 박스-작은 박스 포함형 중복을 한 번 더 제거
            detections = suppressContainedDuplicates(detections);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] 객체 검출 실패: " << error.what() << '\n';
            return 1;
        }

        const auto inferenceEnd = std::chrono::steady_clock::now();

        // 현재 추론 시간에는
        // 전체 프레임 추론 + 원거리 crop 추론 시간이 모두 포함됨
        const double inferenceMilliseconds =
            std::chrono::duration<double, std::milli>(inferenceEnd - inferenceStart).count();

        totalInferenceMilliseconds += inferenceMilliseconds;

        const std::vector<TrackedObject> trackedObjects = tracker.update(detections);

        std::unordered_map<int, ObjectGeometry> geometryById;
        std::unordered_map<int, float> leadScoreById;

        int proposedLeadId = -1;
        float proposedLeadScore = -std::numeric_limits<float>::infinity();

        // 모든 추적 객체의 접지점과 ego lane 내부 위치를 먼저 계산
        for (const TrackedObject& trackedObject : trackedObjects) {
            const cv::Rect& box = trackedObject.box;

            // 차량이 도로에 닿는 위치에 가까운
            // 바운딩 박스 하단 중앙점을 접지점으로 사용
            const cv::Point groundPoint(box.x + box.width / 2, box.y + box.height);

            const bool insideRoad =
                cv::pointPolygonTest(roadRoi, groundPoint, false) >= 0.0;

            const LanePosition lanePosition = calculateLanePosition(egoLaneRoi, groundPoint);

            float leadScore = -std::numeric_limits<float>::infinity();

            if (lanePosition.inside && isVehicleClass(trackedObject.classId)) {
                // 화면 아래쪽에 있을수록 가까운 차량으로 보고 점수 증가
                // 차선 중앙에서 벗어날수록 centerPenalty로 점수 감소
                const float centerPenalty =
                    std::abs(lanePosition.normalizedX - 0.5F) * 90.0F;

                leadScore = static_cast<float>(groundPoint.y) - centerPenalty;

                leadScoreById[trackedObject.trackId] = leadScore;

                if (leadScore > proposedLeadScore) {
                    proposedLeadScore = leadScore;
                    proposedLeadId = trackedObject.trackId;
                }
            }

            geometryById[trackedObject.trackId] = {
                groundPoint,
                insideRoad,
                lanePosition.inside,
                lanePosition.normalizedX,
                leadScore
            };
        }

        const auto activeLeadIterator = leadScoreById.find(activeLeadId);
        const bool activeLeadVisible = activeLeadIterator != leadScoreById.end();

        // 장면 전환 직후에는
        // Tracker와 검출 결과가 다시 안정될 때까지 선행차를 선택하지 않음
        if (!riskAnalysisEnabled) {
            activeLeadId = -1;
        } else if (!activeLeadVisible) {
            activeLeadId = proposedLeadId;
        } else if (proposedLeadId >= 0 &&
                   proposedLeadId != activeLeadId &&
                   proposedLeadScore > activeLeadIterator->second + 35.0F) {
            // 기존 선행차보다 새 후보 점수가 35 이상 높을 때만 선행차를 교체
            activeLeadId = proposedLeadId;
        }

        // 여기부터는 YOLO 검출이 끝난 뒤라
        // ROI와 박스를 영상 위에 그려도 검출 입력에 영향을 주지 않음
        cv::Mat overlay = frame.clone();

        cv::fillConvexPoly(overlay, roadRoi, cv::Scalar(0, 255, 0));
        cv::addWeighted(overlay, 0.10, frame, 0.90, 0.0, frame);

        cv::polylines(frame, roadRoi, true, cv::Scalar(0, 180, 0), 2);
        cv::polylines(frame, egoLaneRoi, true, cv::Scalar(255, 255, 255), 3);

        int objectsOnRoad = 0;
        int objectsInEgoLane = 0;

        RiskResult leadRisk;
        bool leadRiskFound = false;

        for (const TrackedObject& trackedObject : trackedObjects) {
            const auto geometryIterator = geometryById.find(trackedObject.trackId);

            if (geometryIterator == geometryById.end()) {
                continue;
            }

            const ObjectGeometry& geometry = geometryIterator->second;

            if (geometry.insideRoad) {
                ++objectsOnRoad;
            }

            if (geometry.insideEgoLane) {
                ++objectsInEgoLane;
            }

            const bool isLeadTarget = riskAnalysisEnabled &&
                                      trackedObject.trackId == activeLeadId &&
                                      geometry.insideEgoLane;

            const RiskResult rawRisk =
                riskAnalyzer.update(trackedObject, isLeadTarget, processedFrames);

            // 동영상의 멀리 있는 작은 차량에서
            // 박스가 몇 픽셀 흔들린 것만으로 TTC-P가 짧아지는 오경고 방지
            // TTC-P 하나만 보지 않고
            // 샘플 수 / 박스 높이 / 접지점 위치 / 높이 증가 / 접지점 속도를 함께 확인
            const bool hasReliableWarningEvidence =
                isLeadTarget &&
                rawRisk.valid &&
                std::isfinite(rawRisk.ttcSeconds) &&
                rawRisk.sampleCount >= minimumRiskSamplesForWarning &&
                trackedObject.box.height >= minimumLeadBoxHeightForWarning &&
                geometry.groundPoint.y >= minimumLeadGroundYForWarning &&
                rawRisk.heightGrowthRatio >= minimumHeightGrowthRatioForWarning &&
                rawRisk.groundSpeedPixelsPerSecond >= minimumGroundSpeedForWarning;

            RiskResult risk = rawRisk;

            if (risk.level != RiskLevel::Safe && !hasReliableWarningEvidence) {
                risk.level = RiskLevel::Safe;
                risk.valid = false;
                risk.ttcSeconds = std::numeric_limits<float>::infinity();
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

            cv::rectangle(frame, trackedObject.box, boxColor, 3);
            cv::circle(frame, geometry.groundPoint, 6, boxColor, -1);

            std::string label = getClassName(trackedObject.classId) +
                                " ID:" + std::to_string(trackedObject.trackId) +
                                " " + cv::format("%.2f", trackedObject.confidence);

            if (isLeadTarget) {
                label = "LEAD " + label + " " + RiskAnalyzer::toString(risk.level);

                if (risk.valid && std::isfinite(risk.ttcSeconds)) {
                    label += cv::format(" TTC-P:%.1fs", risk.ttcSeconds);
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
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);

            const int labelX = std::clamp(
                trackedObject.box.x,
                0,
                std::max(width - labelSize.width - 10, 0)
            );

            const int labelTop = std::max(trackedObject.box.y, labelSize.height + 10);

            cv::rectangle(
                frame,
                cv::Point(labelX, labelTop - labelSize.height - 10),
                cv::Point(labelX + labelSize.width + 10, labelTop),
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

        // 경고 배너는
        // RiskAnalyzer 결과가 한 번 나왔다고 바로 표시하지 않음
        // 같은 선행 차량에서 CAUTION 8프레임 / DANGER 3프레임 연속 확인 후 표시
        if (!riskAnalysisEnabled || sceneChanged) {
            cautionHoldRemaining   = 0;
            dangerHoldRemaining    = 0;
            cautionCandidateFrames = 0;
            dangerCandidateFrames  = 0;
            warningCandidateLeadId = -1;
        } else if (leadRiskFound && activeLeadId >= 0) {
            // 선행 차량 ID가 바뀌면
            // 이전 차량의 경고 누적 횟수를 새 차량에 넘기지 않음
            if (warningCandidateLeadId != activeLeadId) {
                warningCandidateLeadId = activeLeadId;
                cautionCandidateFrames = 0;
                dangerCandidateFrames  = 0;
            }

            if (leadRisk.level == RiskLevel::Danger) {
                ++dangerCandidateFrames;
                cautionCandidateFrames = 0;

                if (dangerCandidateFrames >= dangerConfirmationFrames) {
                    dangerHoldRemaining  = warningHoldFrames;
                    cautionHoldRemaining = 0;
                }
            } else if (leadRisk.level == RiskLevel::Caution) {
                ++cautionCandidateFrames;
                dangerCandidateFrames = 0;

                if (cautionCandidateFrames >= cautionConfirmationFrames) {
                    cautionHoldRemaining = warningHoldFrames;
                }

                if (dangerHoldRemaining > 0) {
                    --dangerHoldRemaining;
                }
            } else {
                cautionCandidateFrames = 0;
                dangerCandidateFrames  = 0;

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
            dangerCandidateFrames  = 0;

            if (dangerHoldRemaining > 0) {
                --dangerHoldRemaining;
            }

            if (cautionHoldRemaining > 0) {
                --cautionHoldRemaining;
            }
        }

        // 장면 전환 후 위험 분석 유예 시간을 프레임마다 하나씩 감소
        if (sceneWarmupRemaining > 0) {
            --sceneWarmupRemaining;
        }

        koreanText.putText(
            frame,
            "프레임: " + std::to_string(processedFrames),
            cv::Point(30, 40),
            25,
            cv::Scalar(255, 255, 255)
        );

        koreanText.putText(
            frame,
            "추론 시간: " + std::string(cv::format("%.1f ms", inferenceMilliseconds)),
            cv::Point(30, 72),
            25,
            cv::Scalar(255, 255, 255)
        );

        koreanText.putText(
            frame,
            "도로 객체: " + std::to_string(objectsOnRoad) +
                "  내 차선: " + std::to_string(objectsInEgoLane),
            cv::Point(30, 104),
            25,
            cv::Scalar(255, 255, 255)
        );

        std::string leadStatus = "선행 차량: 없음";

        if (activeLeadId >= 0 && leadRiskFound) {
            leadStatus = "선행 차량 ID:" + std::to_string(activeLeadId) +
                         "  상태: " + getRiskNameKorean(leadRisk.level);

            if (leadRisk.valid && std::isfinite(leadRisk.ttcSeconds)) {
                leadStatus += "  TTC-P:" +
                              std::string(cv::format("%.1f초", leadRisk.ttcSeconds));
            } else {
                leadStatus += "  TTC-P:--";
            }
        }

        koreanText.putText(frame, leadStatus, cv::Point(30, 136), 25, cv::Scalar(255, 255, 255));

        if (dangerHoldRemaining > 0) {
            drawCenteredKoreanText(
                koreanText, frame,
                "위험: 충돌 가능성이 높습니다",
                58, 34, cv::Scalar(0, 0, 255)
            );
        } else if (cautionHoldRemaining > 0) {
            drawCenteredKoreanText(
                koreanText, frame,
                "주의: 선행 차량에 접근 중입니다",
                58, 31, cv::Scalar(0, 255, 255)
            );
        }

        writer.write(frame);
    }

    const auto totalEnd = std::chrono::steady_clock::now();

    const double elapsedSeconds =
        std::chrono::duration<double>(totalEnd - totalStart).count();

    const double processingFps =
        elapsedSeconds > 0.0
            ? static_cast<double>(processedFrames) / elapsedSeconds
            : 0.0;

    const double averageInferenceMilliseconds =
        processedFrames > 0
            ? totalInferenceMilliseconds / static_cast<double>(processedFrames)
            : 0.0;

    capture.release();
    writer.release();

    std::cout << '\n';
    std::cout << "[SUCCESS] Step 7 검출 오류 보완 영상 생성 완료\n";
    std::cout << "처리 프레임 수: " << processedFrames << '\n';
    std::cout << "평균 추론 시간: " << std::fixed << std::setprecision(2)
              << averageInferenceMilliseconds << " ms\n";
    std::cout << "전체 처리 속도: " << std::fixed << std::setprecision(2)
              << processingFps << " FPS\n";
    std::cout << "결과 파일: " << outputPath << '\n';

    return 0;
}