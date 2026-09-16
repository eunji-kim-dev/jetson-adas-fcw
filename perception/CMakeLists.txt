add_library(perception_core STATIC
    src/GroupedNms.cpp
    src/InferenceBackendFactory.cpp
    src/MultiObjectTracker.cpp
    src/OpenCVDNNBackend.cpp
    src/VideoFileSource.cpp
    src/YoloCommon.cpp
    src/YoloDetector.cpp
)

target_include_directories(perception_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 헤더에 노출되는 타입 기준으로 PUBLIC/PRIVATE 구분
# - opencv_core   : Detection.hpp / InferenceBackend.hpp의 cv::Rect, cv::Mat
# - opencv_video  : MultiObjectTracker.hpp의 cv::KalmanFilter
# - opencv_dnn    : OpenCVDNNBackend.cpp / GroupedNms.cpp / TensorRTBackend.cpp 구현부에서만 사용
#                   (cv::dnn::Net은 src/OpenCVDNNBackend.hpp 비공개 헤더에만 있음)
# - opencv_imgproc: YoloCommon.cpp의 letterbox(resize/copyMakeBorder)
# - opencv_videoio: VideoFileSource.cpp의 cv::VideoCapture
#                   (헤더에서는 전방 선언만 하므로 공개 의존성이 아님)
target_link_libraries(perception_core
    PUBLIC opencv_core opencv_video
    PRIVATE opencv_imgproc opencv_dnn opencv_videoio
)

# TensorRT 백엔드는 선택 사항임. 최상위에서 ENABLE_TENSORRT=ON 일 때만 붙임
# TensorRT / CUDA 타입은 src/TensorRTBackend.cpp 안에만 있어서 전부 PRIVATE
if(ENABLE_TENSORRT)
    target_sources(perception_core PRIVATE src/TensorRTBackend.cpp)
    target_compile_definitions(perception_core PRIVATE PERCEPTION_HAS_TENSORRT)
    target_include_directories(perception_core PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(perception_core PRIVATE
        ${TENSORRT_LIB}
        ${TENSORRT_ONNXPARSER_LIB}
        CUDA::cudart
    )
endif()