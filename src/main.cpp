#include <opencv2/opencv.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    // 실행할 때 경로 안 넘기면 videos/input.mp4 사용
    const std::string inputPath =
        argc >= 2 ? argv[1] : "videos/input.mp4";

    cv::VideoCapture capture(inputPath);

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
        "results/step3_roi_output.avi";

    // 결과 저장용 writer. 코덱은 MJPG, 나머지는 원본이랑 동일하게
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        sourceFps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr
            << "[ERROR] 결과 영상 파일을 만들 수 없습니다.\n";

        return 1;
    }

    // 차량이 진행할 것으로 예상되는 영역을 사다리꼴로 잡음
    // 좌표를 픽셀 고정값이 아니라 비율로 계산해서
    // 720p든 1080p든 해상도 바뀌어도 그대로 동작함
    const std::vector<cv::Point> drivingRoi = {
        cv::Point(
            static_cast<int>(width * 0.40),
            static_cast<int>(height * 0.60)
        ),
        cv::Point(
            static_cast<int>(width * 0.60),
            static_cast<int>(height * 0.60)
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

    cv::Mat frame;
    int processedFrames = 0;

    // 처리 속도 측정용 시작 시각
    const auto processingStart =
        std::chrono::steady_clock::now();

    while (capture.read(frame)) {
        ++processedFrames;

        // frame에 바로 칠하면 도로가 통째로 가려져서
        // 복사본(overlay)에 먼저 칠하고 나중에 섞는 방식
        cv::Mat overlay = frame.clone();

        // 사다리꼴 내부를 초록으로 채움
        cv::fillConvexPoly(
            overlay,
            drivingRoi,
            cv::Scalar(0, 255, 0)
        );

        // overlay 25% + 원본 75%로 합성 -> ROI가 반투명하게 보임
        cv::addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0.0,
            frame
        );

        // 테두리는 반투명이면 잘 안 보여서 따로 진하게 한 번 더
        cv::polylines(
            frame,
            drivingRoi,
            true,
            cv::Scalar(0, 255, 0),
            3
        );

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

        // ROI 라벨. 사다리꼴 윗변 바로 위에 오도록 비율로 위치 잡음
        cv::putText(
            frame,
            "Driving ROI",
            cv::Point(
                static_cast<int>(width * 0.42),
                static_cast<int>(height * 0.57)
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
        );

        // 다 그린 프레임을 결과 파일에 기록
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
    std::cout
        << "[SUCCESS] ROI 영상 처리가 완료되었습니다.\n";

    std::cout
        << "처리 프레임 수: "
        << processedFrames << '\n';

    std::cout
        << "처리 시간: "
        << std::fixed << std::setprecision(2)
        << elapsedSeconds << "초\n";

    std::cout
        << "처리 속도: "
        << std::fixed << std::setprecision(2)
        << processingFps << " FPS\n";

    std::cout
        << "결과 파일: "
        << outputPath << '\n';

    return 0;
}