// 순수 forward 시간만 측정함
// letterbox·blobFromImage·출력 디코딩·crop 전부 제외함
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

int main(int argc, char** argv) {
    std::string modelPath = (argc > 1) ? argv[1] : "models/yolov8n.onnx";

    std::cout << "OpenCV   : " << CV_VERSION << "\n";
    std::cout << "threads  : " << cv::getNumThreads() << "\n";
    std::cout << "cpus     : " << cv::getNumberOfCPUs() << "\n";
    std::cout << "optimized: " << cv::useOptimized() << "\n\n";

    cv::dnn::Net net = cv::dnn::readNet(modelPath);
    if (net.empty()) {
        std::cerr << "model load failed: " << modelPath << "\n";
        return 1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 288x640 더미 입력 — 실제 영상 대신 고정값을 씀
    cv::Mat blob(std::vector<int>{1, 3, 288, 640}, CV_32F, cv::Scalar(0.5f));

    // 워밍업 10회 — 첫 실행은 초기화가 섞이므로 버림
    for (int i = 0; i < 10; ++i) {
        net.setInput(blob);
        net.forward();
    }

    std::vector<double> ms;
    ms.reserve(100);
    for (int i = 0; i < 100; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        net.setInput(blob);
        cv::Mat out = net.forward();
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();

    std::cout << "--- forward only (1x3x288x640) ---\n";
    std::cout << "mean : " << mean  << " ms\n";
    std::cout << "P50  : " << s[50] << " ms\n";
    std::cout << "P95  : " << s[95] << " ms\n";
    std::cout << "min  : " << s[0]  << " ms\n";
    std::cout << "max  : " << s[99] << " ms\n";
    return 0;
}