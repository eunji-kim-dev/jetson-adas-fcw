add_library(perception_core STATIC
    src/GroupedNms.cpp
    src/InferenceBackendFactory.cpp
    src/LetterboxTables.cpp
    src/MultiObjectTracker.cpp
    src/OpenCVDNNBackend.cpp
    src/ThreadedFrameSource.cpp
    src/V4l2CameraSource.cpp
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
# - opencv_dnn    : OpenCVDNNBackend.cpp / GroupedNms.cpp / TensorRTBackend.cpp / LetterboxTables.cpp 구현부에서만 사용
#                   (LetterboxTables.cpp: GPU 전처리 계수표·CPU 기준 함수. CUDA 없이 컴파일되어 x86 빌드에도 들어감)
#                   (cv::dnn::Net은 src/OpenCVDNNBackend.hpp 비공개 헤더에만 있음)
# - opencv_imgproc: YoloCommon.cpp의 letterbox(resize/copyMakeBorder)
# - opencv_videoio: VideoFileSource.cpp의 cv::VideoCapture
#                   (헤더에서는 전방 선언만 하므로 공개 의존성이 아님)
# - Threads       : ThreadedFrameSource.hpp의 std::thread (헤더에 노출되므로 PUBLIC)
# - V4l2CameraSource.cpp 는 리눅스 헤더(linux/videodev2.h)만 쓰므로 추가 링크 없음
target_link_libraries(perception_core
    PUBLIC opencv_core opencv_video Threads::Threads
    PRIVATE opencv_imgproc opencv_dnn opencv_videoio
)

# TensorRT 백엔드는 선택 사항임. 최상위에서 ENABLE_TENSORRT=ON 일 때만 붙임
# TensorRT / CUDA 타입은 src/TensorRTBackend.cpp 안에만 있어서 전부 PRIVATE
if(ENABLE_TENSORRT)
    # GpuPreprocess.cu: --gpu-preprocess 커널 (nvcc, 최상위에서 enable_language(CUDA))
    target_sources(perception_core PRIVATE src/TensorRTBackend.cpp src/GpuPreprocess.cu)
    set_target_properties(perception_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(perception_core PRIVATE PERCEPTION_HAS_TENSORRT)
    target_include_directories(perception_core PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(perception_core PRIVATE
        ${TENSORRT_LIB}
        ${TENSORRT_ONNXPARSER_LIB}
        CUDA::cudart
    )
endif()