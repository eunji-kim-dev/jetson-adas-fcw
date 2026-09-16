#pragma once

#include <iostream>
#include <string>

/*
 * adas / perception_demo 공통 실행 옵션
 *
 *   [입력 영상 경로]            기본 videos/input.mp4
 *   --backend <name>           기본 opencv_dnn (opencv_dnn | tensorrt_fp32 | tensorrt_fp16)
 *   --run-id <id>              기본 YYYYmmdd-HHMMSS_<영상stem>_<backend> (비우면 RunLogger 가 생성)
 *   --power-mode <str>         기본 unspecified (run_summary.json 기록용)
 *   --warmup-frames <n>        기본 0 (기록만 함, 제외는 분석 스크립트가)
 *   --measured-frames <n>      기본 0 = 끝까지, 아니면 warmup + n 프레임에서 정지
 *   --deadline-ms <ms>         기본 0 = 입력 영상 fps 에서 계산 (1000/fps), 주면 그 값으로 고정
 *   --no-video                 결과 영상을 쓰지 않음 (측정 run 용, 인코딩 부하와 디스크 I/O 제거)
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
};

inline void printUsage(const std::string& programName) {
    std::cerr << "사용법: " << programName
              << " [입력 영상 경로] [--backend NAME] [--run-id ID] [--power-mode MODE]"
              << " [--warmup-frames N] [--measured-frames N]"
              << " [--deadline-ms MS] [--no-video]\n";
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
        } else if (argument.rfind("--", 0) == 0) {
            std::cerr << "[ERROR] 알 수 없는 옵션: " << argument << '\n';
            printUsage(programName);
            return false;
        } else {
            options.inputPath = argument;
        }
    }
    return true;
}
