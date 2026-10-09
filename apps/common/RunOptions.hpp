#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
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
 *                              overlap | history | gap | passby | gate | bonnet | hold | rank | bottom | edge | cropedge | rawconfirm
 *                              rawconfirm: TTC-P 배너의 새 DANGER를 현재 원시 위험 관측 3회로 확정함. 이미 확정한 경고의 안정화 유지와 CAUTION은 기존 정책을 씀
 *                              생략하면 기존 규칙 그대로 (golden 보존). adas 전용, perception_demo 는 무시함
 *   --ttc-mode <모드>           proxy (기본, golden 보존) | homography | both. TTR-H 실험 (5-2). adas 전용, 영상 파일 입력만 (--camera·--threaded-capture 불가)
 *                              homography: 배너를 TTR-H 분기로만 띄움 / both: P·H 를 합치지 않고 각각 기록, 배너는 P
 *   --road-points <8개 숫자>    노면 4점 픽셀 좌표 (가까운 왼쪽, 가까운 오른쪽, 먼 오른쪽, 먼 왼쪽 순서의 x,y. videos.csv rx1..ry4)
 *   --road-size <W>,<L>        노면 4점 사각형의 폭·길이 (m, 예: 3.5,10). TTR-H 시간에서는 약분됨
 *   --road-res <W>x<H>         노면 4점을 찍은 해상도 (가로x세로, videos.csv road_res). 영상 크기와 다르면 실행을 막음
 *   --bonnet-y <y>             보닛 선 y. 생략하면 화면 맨 아래 줄을 기준으로 씀
 *   --road-frame <n>           노면 4점을 찍은 프레임 (기록용)
 *   --road-status <글자>        assumed | verified (기록용)
 *   --d0-m <m>                 범퍼~가시 노면 끝 여유 (예약 값, 계산에 안 씀. 기록용)
 *                              --road-* · --bonnet-y · --d0-m 은 --ttc-mode homography|both 와 같이 줘야 함
 *   --gpu-preprocess           tensorrt_* 전용. 전처리(letterbox·RGB·/255)를 GPU 커널 한 번으로 함 (기본 꺼짐 = CPU 경로, golden 보존)
 *   --gpu-preprocess-check     --gpu-preprocess 와 같이 돌면서 프레임마다 GPU 입력 텐서를 CPU 경로와 비트 단위로 비교함
 *                              끝날 때 표준 에러로 요약을 냄 (확인용이라 느림. 측정에는 안 씀)
 *   --camera-zero-copy         --camera 전용. GPU 전처리 v2: 카메라 MMAP 버퍼(YUYV)를 CUDA 에 등록해 GPU 가 복사 없이 직접 읽음
 *                              --gpu-preprocess 를 같이 켬. --gpu-preprocess-check 와 같이 주면 v2 입력 텐서를 CPU 경로와 비교함
 *                              --threaded-capture 와는 같이 못 씀 (버퍼를 다음 프레임까지 들고 있어야 함)
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
    // --ttc-mode homography|both (TTR-H 실험). proxy 면 아래 노면 값은 비어 있음
    std::string ttcMode = "proxy";       // proxy | homography | both
    std::vector<double> roadPointsPx;    // --road-points 8개 (P1..P4 의 x,y). 비어 있으면 없음
    double roadWidthM = 0.0;             // --road-size 의 W (m). 0 이면 없음
    double roadLengthM = 0.0;            // --road-size 의 L (m)
    int roadResWidth = 0;                // --road-res 의 가로. 0 이면 없음
    int roadResHeight = 0;               // --road-res 의 세로
    std::optional<int> bonnetY;          // --bonnet-y. 없으면 화면 맨 아래 줄 기준
    std::optional<int> roadFrame;        // --road-frame (기록용)
    std::string roadStatus;              // --road-status (기록용)
    std::optional<double> d0M;           // --d0-m (예약, 기록용)
    bool gpuPreprocess = false;          // --gpu-preprocess: TensorRT 입력을 GPU 커널로 만듦
    bool gpuPreprocessCheck = false;     // --gpu-preprocess-check: GPU 입력 텐서를 CPU 경로와 비교 (gpuPreprocess 도 켬)
    bool cameraZeroCopy = false;         // --camera-zero-copy: 카메라 YUYV 버퍼를 GPU 가 직접 읽음 (GPU 전처리 v2, gpuPreprocess 도 켬)
};

// --lead-rule 에 쓸 수 있는 이름. 순서는 문서·로그 표기 순서와 같음
inline const std::vector<std::string>& leadRuleNames() {
    static const std::vector<std::string> names = {"overlap", "history", "gap", "passby", "gate", "bonnet", "hold", "rank", "bottom", "edge", "cropedge", "rawconfirm"};
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
              << " [--lead-rule overlap,history,gap,passby,gate,bonnet,hold,rank,bottom,edge,cropedge,rawconfirm]"
              << " [--ttc-mode proxy|homography|both --road-points x1,y1,...,x4,y4 --road-size W,L"
              << " [--road-res WxH] [--bonnet-y Y] [--road-frame N] [--road-status TEXT] [--d0-m M]]"
              << " [--gpu-preprocess | --gpu-preprocess-check] [--camera-zero-copy]\n";
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
            // 뒤에 글자가 붙은 값(850abc)은 거부함. stoi 는 앞 숫자만 읽고 넘어감
            std::size_t used = 0;
            out = std::stoi(text, &used);
            if (used != text.size()) throw std::invalid_argument(text);
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
            std::size_t used = 0;
            out = std::stod(text, &used);
            if (used != text.size() || !std::isfinite(out)) throw std::invalid_argument(text);
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

    // 콤마로 이은 숫자 목록. 개수가 다르거나 숫자가 아닌 칸(빈 칸·뒤에 글자 붙은 칸·nan·inf 포함)이 있으면 실패로 처리함
    auto parseNumberList = [&](const std::string& option, const std::string& text, std::size_t expectedCount, std::vector<double>& out) {
        std::vector<double> values;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = text.find(',', start);
            const std::string token = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            try {
                std::size_t used = 0;
                values.push_back(std::stod(token, &used));
                if (used != token.size() || !std::isfinite(values.back())) throw std::invalid_argument(token);
            } catch (const std::exception&) {
                std::cerr << "[ERROR] " << option << " 값이 숫자가 아님: '" << token << "'\n";
                return false;
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (values.size() != expectedCount) {
            std::cerr << "[ERROR] " << option << " 는 숫자 " << expectedCount << "개여야 함, 받은 개수: " << values.size() << '\n';
            return false;
        }
        out = values;
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
                    std::cerr << "[ERROR] --lead-rule 에 모르는 이름: '" << token << "' (가능: overlap,history,gap,passby,gate,bonnet,hold,rank,bottom,edge,cropedge,rawconfirm)\n";
                    return false;
                }
                if (std::find(options.leadRules.begin(), options.leadRules.end(), token) == options.leadRules.end()) {
                    options.leadRules.push_back(token);
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
        } else if (argument == "--gpu-preprocess") {
            options.gpuPreprocess = true;
        } else if (argument == "--gpu-preprocess-check") {
            options.gpuPreprocess = true;
            options.gpuPreprocessCheck = true;
        } else if (argument == "--camera-zero-copy") {
            options.gpuPreprocess = true;
            options.cameraZeroCopy = true;
        } else if (argument == "--ttc-mode") {
            if (!takeValue(i, argument, options.ttcMode)) return false;
            if (options.ttcMode != "proxy" && options.ttcMode != "homography" && options.ttcMode != "both") {
                std::cerr << "[ERROR] --ttc-mode 는 proxy | homography | both: " << options.ttcMode << '\n';
                return false;
            }
        } else if (argument == "--road-points") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            if (!parseNumberList(argument, text, 8, options.roadPointsPx)) return false;
        } else if (argument == "--road-size") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            std::vector<double> size;
            if (!parseNumberList(argument, text, 2, size)) return false;
            if (!(size[0] > 0.0) || !(size[1] > 0.0)) {
                std::cerr << "[ERROR] --road-size 값은 0 보다 커야 함: " << text << '\n';
                return false;
            }
            options.roadWidthM = size[0];
            options.roadLengthM = size[1];
        } else if (argument == "--road-res") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            // "1920x1080" 처럼 가로x세로 (videos.csv road_res 그대로. --crop-input 의 세로x가로와 순서가 다름)
            const std::size_t split = text.find('x');
            try {
                if (split == std::string::npos) throw std::invalid_argument(text);
                const std::string widthText = text.substr(0, split);
                const std::string heightText = text.substr(split + 1);
                std::size_t widthUsed = 0;
                std::size_t heightUsed = 0;
                options.roadResWidth = std::stoi(widthText, &widthUsed);
                options.roadResHeight = std::stoi(heightText, &heightUsed);
                if (widthUsed != widthText.size() || heightUsed != heightText.size()) throw std::invalid_argument(text);
            } catch (const std::exception&) {
                std::cerr << "[ERROR] --road-res 는 WxH 형식이어야 함 (예: 1920x1080): " << text << '\n';
                return false;
            }
            if (options.roadResWidth <= 0 || options.roadResHeight <= 0) {
                std::cerr << "[ERROR] --road-res 값은 0 보다 커야 함: " << text << '\n';
                return false;
            }
        } else if (argument == "--bonnet-y") {
            int value = 0;
            if (!takeInt(i, argument, value)) return false;
            options.bonnetY = value;
        } else if (argument == "--road-frame") {
            int value = 0;
            if (!takeInt(i, argument, value)) return false;
            options.roadFrame = value;
        } else if (argument == "--road-status") {
            if (!takeValue(i, argument, options.roadStatus)) return false;
        } else if (argument == "--d0-m") {
            std::string text;
            if (!takeValue(i, argument, text)) return false;
            std::vector<double> value;
            if (!parseNumberList(argument, text, 1, value)) return false;
            if (value[0] < 0.0) {
                std::cerr << "[ERROR] --d0-m 값은 0 이상이어야 함: " << text << '\n';
                return false;
            }
            options.d0M = value[0];
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
    // TTR-H: homography|both 는 노면 4점·크기가 있어야 돎. proxy 에 노면 옵션을 주면 안 쓰이므로 막음 (TTR-H 없이 조용히 돌지 않게)
    const bool hasRoadOption = !options.roadPointsPx.empty() || options.roadWidthM > 0.0 || options.roadResWidth > 0
        || options.bonnetY.has_value() || options.roadFrame.has_value() || !options.roadStatus.empty() || options.d0M.has_value();
    if (options.ttcMode == "proxy") {
        if (hasRoadOption) {
            std::cerr << "[ERROR] --road-* / --bonnet-y / --d0-m 는 --ttc-mode homography 또는 both 와 같이 줘야 함\n";
            return false;
        }
    } else {
        if (options.roadPointsPx.empty() || !(options.roadWidthM > 0.0)) {
            std::cerr << "[ERROR] --ttc-mode " << options.ttcMode << " 는 --road-points 와 --road-size 가 있어야 함\n";
            printUsage(programName);
            return false;
        }
        // 거리 샘플 시간을 프레임 번호 ÷ fps 로 계산하므로 이번 실험은 영상 파일만 받음 (5-2)
        if (!options.cameraDevice.empty()) {
            std::cerr << "[ERROR] --ttc-mode " << options.ttcMode << " 는 카메라 입력을 지원하지 않음 (영상 파일만)\n";
            return false;
        }
        // 캡처 스레드 분리는 최신 프레임 우선이라 영상 파일에서도 프레임을 건너뜀 → 프레임 번호 ÷ fps 시간이 어긋남
        if (options.threadedCapture) {
            std::cerr << "[ERROR] --ttc-mode " << options.ttcMode << " 는 --threaded-capture 와 같이 못 씀 (건너뛴 프레임 때문에 거리 샘플 시간이 어긋남)\n";
            return false;
        }
    }
    // GPU 전처리는 TensorRT 백엔드에만 있음 (opencv_dnn 은 CPU 추론이라 입력을 GPU 로 만들 이유가 없음)
    if (options.gpuPreprocess && options.backendName.rfind("tensorrt_", 0) != 0) {
        std::cerr << "[ERROR] --gpu-preprocess 는 tensorrt_* 백엔드에서만 씀: " << options.backendName << '\n';
        return false;
    }
    // GPU 전처리 v2 는 카메라 MMAP 버퍼를 처리가 끝날 때(다음 read())까지 들고 있어야 함
    if (options.cameraZeroCopy) {
        if (options.cameraDevice.empty()) {
            std::cerr << "[ERROR] --camera-zero-copy 는 --camera 와 같이 줘야 함 (영상 파일에는 카메라 버퍼가 없음)\n";
            return false;
        }
        // 캡처 스레드는 처리 중에 다음 프레임을 읽어 들고 있던 버퍼를 드라이버에 돌려줌 → GPU 가 읽는 중에 덮어써질 수 있음
        if (options.threadedCapture) {
            std::cerr << "[ERROR] --camera-zero-copy 는 --threaded-capture 와 같이 못 씀 (처리 중에 카메라 버퍼가 돌아감)\n";
            return false;
        }
    }
    return true;
}
