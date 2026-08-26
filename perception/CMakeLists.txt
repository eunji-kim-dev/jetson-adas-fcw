add_library(perception_core STATIC
    src/GroupedNms.cpp
    src/InferenceBackendFactory.cpp
    src/MultiObjectTracker.cpp
    src/OpenCVDNNBackend.cpp
    src/VideoFileSource.cpp
    src/YoloDetector.cpp
)

target_include_directories(perception_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 헤더에 노출되는 타입 기준으로 PUBLIC/PRIVATE 구분
# - opencv_core   : Detection.hpp / InferenceBackend.hpp의 cv::Rect, cv::Mat
# - opencv_video  : MultiObjectTracker.hpp의 cv::KalmanFilter
# - opencv_dnn    : OpenCVDNNBackend.cpp / GroupedNms.cpp 구현부에서만 사용
#                   (cv::dnn::Net은 src/OpenCVDNNBackend.hpp 비공개 헤더에만 있음)
# - opencv_imgproc: OpenCVDNNBackend.cpp의 letterbox(resize/copyMakeBorder)
# - opencv_videoio: VideoFileSource.cpp의 cv::VideoCapture
#                   (헤더에서는 전방 선언만 하므로 공개 의존성이 아님)
target_link_libraries(perception_core
    PUBLIC opencv_core opencv_video
    PRIVATE opencv_imgproc opencv_dnn opencv_videoio
)
