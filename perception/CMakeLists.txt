add_library(perception_core STATIC
    src/MultiObjectTracker.cpp
    src/YoloDetector.cpp
)

target_include_directories(perception_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 헤더에 노출되는 타입 기준으로 PUBLIC/PRIVATE 구분
# - opencv_core : Detection.hpp의 cv::Rect
# - opencv_video: MultiObjectTracker.hpp의 cv::KalmanFilter
# - opencv_dnn  : YoloDetector.hpp의 cv::dnn::Net
target_link_libraries(perception_core
    PUBLIC opencv_core opencv_video opencv_dnn
    PRIVATE opencv_imgproc
)
