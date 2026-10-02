#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
 * adas / perception_demo 공통 실행 옵션
 *
 *   [입력 영상 경로]            기본 videos/input.mp4
 *   --backend <name>           기본 opencv_dnn (opencv_dnn | tensorrt_fp32 | tensorrt_fp16 | tensorrt_int8)
 *   --run-id <id>              기본 YYYYmmdd-HHMMSS_<영상stem>_<backend> (비우면 RunLogger 가 생성)
 *   --power-mode <str>         기본 unspecified (run_summary.json 기록용)
 *   --warmup-frames <n>        기본 0 (기록만 함, 제외는 분석 스크립트가)
 *   --measured-frames <n>      기본 0 = 끝까지, 아니면 warmup + n 프레임에서 정지
 *   --deadline-ms <ms>         기본 0 = 입력 영상 fps 에서 계산 (1000/fps), 주면 그 값으로 고정
 *   --no-video                 결과 영상을 쓰지 않음 (측정 run 용, 인코딩 부하와 디스크 I/O 제거)
 *   --lead-select-log          LEAD 선정 근거 CSV (results/<id>_lead_select.csv). 골든 파일과 별개
 *   --diag-log                 TTC-P·배너 판정 사슬 진단 CSV (results/<id>_diag.csv). 골든 파일과 별개
 *   --lane-roi <8개 정수>       영상별 ego lane ROI 를 원본 픽셀 좌표로 지정 (TL,TR,BR,BL 순서, 콤마 구분)
 *                              예: --lane-roi 854,520,941,520,1257,925,198,925
 *                              생략하면 기존 화면 비율 ROI 를 그대로 씀 (golden 보존)
 *   --crop-model <경로>         원거리 crop 추론에 쓸 ONNX (예: models/yolov8n_288x640.onnx)
 *   --crop-input <H>x<W>       그 모델의 입력 크기. 세로x가로 순서 (예: 288x640). --crop-model 과 같이 줘야 함
 *                              둘 다 생략하면 crop 도 전체 프레임과 같은 640x640 모델로 추론함 (golden 보존)
 *   --calib-list <txt>         tensorrt_int8 전용. 전체 프레임 모델의 calibration 이미지 목록 (한 줄에 경로 하나)
 *   --crop-calib-list <txt>    tensorrt_int8 + --crop-model 전용. crop 모델의 calibration 이미지 목록
 *                              둘 다 int8 엔진을 처음 만들 때만 필요함. 엔진·캐시가 있으면 생략 가능
 *   --camera <장치>            입력을 영상 파일 대신 V4L2 카메라로 (예: /dev/video0). 640x480 YUYV 30fps 로 열고 자동 노출을 끔
 *                              스트림이 끝나지 않으므로 --measured-frames 로 멈춤. 준비 구간은 --warmup-frames 40 으로 분석에서 뺌
 *   --threaded-capture         캡처 스레드를 분리하고 최신 프레임만 처리함 (기본 꺼짐 = 기존 동기 경로, golden 보존)
 *   --int8-variant <이름>       tensorrt_int8 실험용. 엔진·calibration 캐시를 <모델>.int8-<이름>.* 로 따로 만듦 (정식 엔진 보존)
 *   --int8-calibrator <종류>    entropy (기본) | minmax. --int8-variant 와 같이 줘야 함
 *   --int8-shuffle-seed <n>    calibration 이미지 순서를 시드 n 으로 섞음. --int8-variant 와 같이 줘야 함
 *   --int8-fp32-head           검출 헤드(/model.22/) 층을 FP32 로 강제. --int8-variant 와 같이 줘야 함
 *   --lead-rule <목록>          LEAD 선택·경고 게이트 규칙 플래그. 쉼표로 이어 줌 (예: --lead-rule overlap,gap)
 *                              overlap | history | gap | passby | gate | bonnet | hold (뜻은 adas/LeadSelector.hpp 의 LeadRuleFlags 참고)
 *                              생략하면 기존 규칙 그대로 (golden 보존). adas 전용, perception_demo 는 무시함
 */

struct RunOptions {
    std::string inputPath = "videos/input.mp4";
    std::string backendName = "opencv_dnn";
    std::string runId;
    std::string powerMode = "unspecified";
    int warmupFrames = 0;
    int measuredFrames = 0;
    double deadlineMs = 0.0;   // 0 이면 영상 fps 기준
    bool writeVideo = true;
    bool leadSelectLog = false;   // --lead-select-log: 프레임·트랙별 LEAD 선정 근거를 results/<id>_lead_select.csv 에 씀
    bool diagLog = false;         // --diag-log: 프레임·트랙별 TTC-P·배너 판정 사슬을 results/<id>_diag.csv 에 씀
    std::vector<int> laneRoiPx;   // 비어 있으면 기본 비율 ROI, 아니면 8개 (x1,y1,...,x4,y4)
    std::string cropModelPath;    // 비어 있으면 crop 도 전체 프레임 모델을 씀
    int cropInputHeight = 0;      // --crop-input 의 H. 0 이면 미지정
    int cropInputWidth = 0;       // --crop-input 의 W. 0 이면 미지정
    std::string calibList;        // int8 전용. 전체 프레임 모델 calibration 이미지 목록
    std::string cropCalibList;    // int8 전용. crop 모델 calibration 이미지 목록
    std::string cameraDevice;     // 비어 있으면 영상 파일(inputPath), 아니면 V4L2 장치 경로
    bool threadedCapture = false; // true 면 캡처 스레드 분리 (최신 프레임 우선)
    std::string int8Variant;      // int8 실험 이름. 비어 있으면 정식 엔진
    std::string int8Calibrator = "entropy";  // entropy | minmax
    int int8ShuffleSeed = -1;     // 0 이상이면 calibration 순서 섞음
    bool int8Fp32Head = false;    // 헤드 FP32 강제
    std::vector<std::string> leadRules;  // --lead-rule 이름 목록. 비어 있으면 기존 규칙
};

// --lead-rule 에 쓸 수 있는 이름. 순서는 문서·로그 표기 순서와 같음
inline const std::vector<std::string>& leadRuleNames() {
    static const std::vector<std::string> names = {"overlap", "history", "gap", "passby", "gate", "bonnet", "hold", "rank"};
    return names;
}

inline void printUsage(const std::string& programName) {
    std::cerr << "사용법: " << programName
              << " [입력 영상 경로] [--backend NAME] [--run-id ID] [--power-mode MODE]"
              << " [--warmup-frames N] [--measured-frames N]"
              << " [--deadline-ms MS] [--no-video] [--diag-log] [--lane-roi x1,y1,x2,y2,x3,y3,x4,y4]"
              << " [--crop-model PATH --crop-input HxW]"
              << " [--calib-list TXT] [--crop-calib-list TXT]"
              << " [--camera /dev/videoN] [--threaded-capture]"
              << " [--int8-variant NAME [--int8-calibrator entropy|minmax] [--int8-shuffle-seed N] [--int8-fp32-head]]"
              << " [--lead-rule overlap,history,gap,passby,gate,bonnet,hold,rank]\n";
}

// 실패하면 false 를 돌려주고 이유를 stderr 에 출력
inline bool parseRunOptions(int argc, char* argv[], const std::string& programName, RunOptions& options) {
    auto takeValue = [&](int& i, const std::string& option, std::string& out) {
        if (i + 1 >= argc) {
            std::cerr << "[ERROR] " << option << " 옵션에 값이 없음\n";
            printUsage(programName);
            return false;
        }
        out = argv[++i];
        return true;
    };
    auto takeInt = [&](int& i, const std::string& option, int& out) {
        std::string text;
        if (!takeValue(i, option, text)) return false;
        try {
            out = std::stoi(text);
        } catch (const std::exception&) {
            std::cerr << "[ERROR] " << option << " 값이 정수가 아님: " << text << '\n';
            return false;
        }
        if (out < 0) {
            std::cerr << "[ERROR] " << option << " 값은 0 이상이어야 함: " << text << '\n';
            return false;
        }
        return true;
    };

    auto takeDouble = [&](int& i, const std::string& option, double& out) {
        std::string text;
        if (!takeValue(i, option, text)) return false;
        try {
            out = std::stod(text);
        } catch (const std::exception&) {
            std::cerr << "[ERROR] " << option << " 값이 숫자가 아님: " << text << '\n';
            return false;
        }
        if (out <= 0.0) {
            std::cerr << "[ERROR] " << option << " 값은 0 보다 커야 함: " << text << '\n';
            return false;
        }
        return true;
    };    

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--backend") {
            if (!takeValue(i, argument, options.backendName)) return false;
        } else if (argument.rfind("--backend=", 0) == 0) {
            options.backendName = argument.substr(std::string("--backend=").size());
        } else if (argument == "--run-id") {
            if (!takeValue(i, argument, options.runId)) return false;
        } else if (argument == "--power-mode") {
            if (!takeValue(i, argument, options.powerMode)) return false;
        } else if (argument == "--warmup-frames") {
            if (!takeInt(i, argument, options.warmupFrames)) return false;
        } else if (argument == "--measured-frames") {
            if (!takeInt(i, argument, options.measuredFrames)) return false;
        } else if (argument == "--deadline-ms") {
            if (!takeDouble(i, argument, options.deadlineMs)) return false;
        } else if (argument == "--no-video") {
            options.writeVideo = false;
        } else if (argument == "--lead-select-log") {
            options.leadSelectLog = true;
        } else if (argument == "--diag-log") {
            options.diagLog = true;
        } else if (argument == "--lane-roi") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            // 콤마로 잘라 정수 8개로 읽음. 개수가 다르거나 숫자가 아니면 실패로 처리함
            std::vector<int> values;
            std::stringstream stream(text);
            std::string token;
            while (std::getline(stream, token, ',')) {
                try {
                    values.push_back(std::stoi(token));
                } catch (const std::exception&) {
                    std::cerr << "[ERROR] --lane-roi 값이 정수가 아님: " << token << '\n';
                    return false;
                }
            }
            if (values.size() != 8) {
                std::cerr << "[ERROR] --lane-roi 는 정수 8개(x1,y1,...,x4,y4)여야 함, 받은 개수: " << values.size() << '\n';
                return false;
            }
            options.laneRoiPx = values;
        } else if (argument == "--calib-list") {
            if (!takeValue(i, argument, options.calibList)) return false;
        } else if (argument == "--crop-calib-list") {
            if (!takeValue(i, argument, options.cropCalibList)) return false;
        } else if (argument == "--camera") {
            if (!takeValue(i, argument, options.cameraDevice)) return false;
        } else if (argument == "--threaded-capture") {
            options.threadedCapture = true;
        } else if (argument == "--int8-variant") {
            if (!takeValue(i, argument, options.int8Variant)) return false;
        } else if (argument == "--int8-calibrator") {
            if (!takeValue(i, argument, options.int8Calibrator)) return false;
            if (options.int8Calibrator != "entropy" && options.int8Calibrator != "minmax") {
                std::cerr << "[ERROR] --int8-calibrator 는 entropy 또는 minmax: " << options.int8Calibrator << '\n';
                return false;
            }
        } else if (argument == "--int8-shuffle-seed") {
            if (!takeInt(i, argument, options.int8ShuffleSeed)) return false;
        } else if (argument == "--int8-fp32-head") {
            options.int8Fp32Head = true;
        } else if (argument == "--lead-rule") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            // 쉼표로 잘라 이름을 확인함. 모르는 이름·빈 항목(빈 값, 끝 쉼표, 연속 쉼표)이면 실패로 처리함 (오타로 조용히 기존 규칙이 돌지 않게)
            // std::getline 은 끝 쉼표 뒤의 빈 항목을 안 돌려주므로 직접 자름. 빈 값 검사는 이번에 받은 text 기준임
            if (text.empty()) {
                std::cerr << "[ERROR] --lead-rule 값이 비어 있음\n";
                return false;
            }
            std::size_t start = 0;
            while (true) {
                const std::size_t comma = text.find(',', start);
                const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (token.empty()) {
                    std::cerr << "[ERROR] --lead-rule 에 빈 항목이 있음 (쉼표 앞뒤가 비었음): '" << text << "'\n";
                    return false;
                }
                const auto& names = leadRuleNames();
                if (std::find(names.begin(), names.end(), token) == names.end()) {
                    std::cerr << "[ERROR] --lead-rule 에 모르는 이름: '" << token << "' (가능: overlap,history,gap,passby,gate,bonnet,hold,rank)\n";
                    return false;
                }
                if (std::find(options.leadRules.begin(), options.leadRules.end(), token) == options.leadRules.end()) {
                    options.leadRules.push_back(token);
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        } else if (argument == "--crop-model") {
            if (!takeValue(i, argument, options.cropModelPath)) return false;
        } else if (argument == "--crop-input") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            // "288x640" 처럼 세로x가로. YOLO imgsz 와 ONNX 파일 이름 순서에 맞춤
            const std::size_t split = text.find('x');
            if (split == std::string::npos) {
                std::cerr << "[ERROR] --crop-input 은 HxW 형식이어야 함 (예: 288x640): " << text << '\n';
                return false;
            }
            try {
                options.cropInputHeight = std::stoi(text.substr(0, split));
                options.cropInputWidth = std::stoi(text.substr(split + 1));
            } catch (const std::exception&) {
                std::cerr << "[ERROR] --crop-input 값이 정수가 아님: " << text << '\n';
                return false;
            }
            if (options.cropInputHeight <= 0 || options.cropInputWidth <= 0) {
                std::cerr << "[ERROR] --crop-input 값은 0 보다 커야 함: " << text << '\n';
                return false;
            }
        } else if (argument.rfind("--", 0) == 0) {
            std::cerr << "[ERROR] 알 수 없는 옵션: " << argument << '\n';
            printUsage(programName);
            return false;
        } else {
            options.inputPath = argument;
        }
    }

    // --crop-model 과 --crop-input 은 짝으로만 씀. 하나만 있으면 어떤 크기로 돌릴지 알 수 없음
    const bool hasCropModel = !options.cropModelPath.empty();
    const bool hasCropInput = options.cropInputHeight > 0 && options.cropInputWidth > 0;
    if (hasCropModel != hasCropInput) {
        std::cerr << "[ERROR] --crop-model 과 --crop-input 은 같이 줘야 함\n";
        printUsage(programName);
        return false;
    }
    // int8 실험 옵션은 이름 없이 쓰면 정식 엔진·캐시 파일을 덮어쓰므로 막음
    const bool hasInt8Experiment = options.int8Calibrator != "entropy" || options.int8ShuffleSeed >= 0 || options.int8Fp32Head;
    if (hasInt8Experiment && options.int8Variant.empty()) {
        std::cerr << "[ERROR] --int8-calibrator / --int8-shuffle-seed / --int8-fp32-head 는 --int8-variant 이름과 같이 줘야 함 (정식 엔진 보호)\n";
        return false;
    }
    return true;
}
