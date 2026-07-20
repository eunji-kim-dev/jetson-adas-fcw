#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

int main() {
    const int width = 640;
    const int height = 360;

    cv::Mat image(
        height,
        width,
        CV_8UC3,
        cv::Scalar(30, 30, 30)
    );

    cv::putText(
        image,
        "OpenCV C++ is working!",
        cv::Point(80, 190),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 255, 0),
        2
    );

    const std::string outputPath =
        "results/opencv_test.png";

    const bool saved =
        cv::imwrite(outputPath, image);

    if (!saved) {
        std::cerr
            << "[ERROR] 이미지를 저장하지 못했습니다.\n";

        return 1;
    }

    std::cout
        << "[SUCCESS] OpenCV 테스트가 완료되었습니다.\n";

    std::cout
        << "결과 파일: " << outputPath << '\n';

    return 0;
}