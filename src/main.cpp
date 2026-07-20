#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // 실행할 때 경로 안 넘기면 videos/input.mp4 사용
    const std::string inputPath =
        argc >= 2 ? argv[1] : "videos/input.mp4";

    // 영상 열기
    cv::VideoCapture capture(inputPath);

    // 못 열면 여기서 끝냄
    if (!capture.isOpened()) {
        std::cerr
            << "[ERROR] 영상을 열 수 없습니다: "
            << inputPath << '\n';

        return 1;
    }

    // 원본 영상 정보 (해상도, 프레임 수, fps)
    const int width = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_WIDTH)
    );

    const int height = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    const int totalFrames = static_cast<int>(
        capture.get(cv::CAP_PROP_FRAME_COUNT)
    );

    double sourceFps =
        capture.get(cv::CAP_PROP_FPS);

    // 영상에 따라 fps가 0으로 나오는 경우가 있어서 그때는 30으로 가정
    if (sourceFps <= 0.0) {
        sourceFps = 30.0;
    }

    std::cout << "입력 영상: " << inputPath << '\n';
    std::cout << "해상도: "
              << width << " x " << height << '\n';
    std::cout << "원본 FPS: " << sourceFps << '\n';
    std::cout << "전체 프레임 수: "
              << totalFrames << '\n';

    const std::string outputPath =
        "results/step2_output.avi";

    // 결과 저장용 writer. 코덱은 MJPG, 나머지는 원본이랑 동일하게
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        sourceFps,
        cv::Size(width, height)
    );

    // results 폴더가 없거나 코덱 문제 있으면 여기서 걸림
    if (!writer.isOpened()) {
        std::cerr
            << "[ERROR] 결과 영상 파일을 만들 수 없습니다.\n";

        return 1;
    }

    cv::Mat frame;
    int processedFrames = 0;

    // 처리 속도 측정용 시작 시각
    const auto processingStart =
        std::chrono::steady_clock::now();

    // 프레임 단위로 읽어서 처리. read()가 false 리턴하면 영상 끝
    while (capture.read(frame)) {
        ++processedFrames;

        // 프레임에 얹을 텍스트
        const std::string frameText =
            "Frame: " + std::to_string(processedFrames);

        const std::string fpsText =
            "Source FPS: " +
            std::to_string(static_cast<int>(sourceFps));

        // 프레임 번호 (초록, 좌상단)
        cv::putText(
            frame,
            frameText,
            cv::Point(30, 50),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        // 원본 fps (노랑, 그 밑에)
        cv::putText(
            frame,
            fpsText,
            cv::Point(30, 90),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 255),
            2
        );

        // 텍스트 얹은 프레임을 결과 파일에 기록
        writer.write(frame);
    }

    const auto processingEnd =
        std::chrono::steady_clock::now();

    // 전체 걸린 시간(초)
    const double elapsedSeconds =
        std::chrono::duration<double>(
            processingEnd - processingStart
        ).count();

    // 초당 몇 프레임 처리했는지. 0으로 나누기만 피함
    const double processingFps =
        elapsedSeconds > 0.0
            ? processedFrames / elapsedSeconds
            : 0.0;

    capture.release();
    writer.release();

    // 처리 결과 요약 출력
    std::cout << '\n';
    std::cout << "[SUCCESS] 영상 처리가 완료되었습니다.\n";
    std::cout << "처리 프레임 수: "
              << processedFrames << '\n';

    std::cout
        << "처리 시간: "
        << std::fixed << std::setprecision(2)
        << elapsedSeconds << "초\n";

    std::cout
        << "처리 속도: "
        << std::fixed << std::setprecision(2)
        << processingFps << " FPS\n";

    std::cout << "결과 파일: "
              << outputPath << '\n';

    return 0;
}