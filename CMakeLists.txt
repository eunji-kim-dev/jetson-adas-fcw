# 최상위 CMakeLists.txt
# 프로젝트 공통 설정만 정의하고,
# perception / adas / apps의 세부 빌드 설정은 각 디렉터리에 위임함

# 3.17 부터 find_package(CUDAToolkit) 을 쓸 수 있음
cmake_minimum_required(VERSION 3.17)
project(jetson_adas_fcw CXX)

# 프로젝트 전체에서 C++17을 사용함
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 한글 오버레이에 사용하는 OpenCV freetype 모듈을 선택적으로 사용함
# 기본값은 ON이며, freetype이 없는 환경에서는
# -DENABLE_FREETYPE=OFF로 빌드할 수 있음
option(ENABLE_FREETYPE "opencv_freetype 기반 한글 오버레이 사용" ON)

# TensorRT 백엔드 (tensorrt_fp32 / tensorrt_fp16) 를 빌드에 넣을지
# 기본값은 OFF 라 x86 노트북에서는 아무것도 안 바뀜
# Jetson 에서는 -DENABLE_TENSORRT=ON 으로 켬
option(ENABLE_TENSORRT "TensorRT 추론 백엔드 사용" OFF)

if(ENABLE_FREETYPE)
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        videoio
        dnn
        video
        freetype
    )
else()
    find_package(OpenCV REQUIRED COMPONENTS
        core
        imgproc
        videoio
        dnn
        video
    )
endif()

if(ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)

    # TensorRT 는 공식 CMake 설정 파일이 없어서 직접 찾음
    # JetPack: /usr/include/aarch64-linux-gnu, /usr/lib/aarch64-linux-gnu
    # x86 tar 설치: -DTENSORRT_ROOT=/opt/TensorRT-10.x 로 알려줌
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT}/include
              /usr/include/aarch64-linux-gnu
              /usr/include/x86_64-linux-gnu
              /usr/include
    )
    find_library(TENSORRT_LIB nvinfer
        HINTS ${TENSORRT_ROOT}/lib
              /usr/lib/aarch64-linux-gnu
              /usr/lib/x86_64-linux-gnu
    )
    find_library(TENSORRT_ONNXPARSER_LIB nvonnxparser
        HINTS ${TENSORRT_ROOT}/lib
              /usr/lib/aarch64-linux-gnu
              /usr/lib/x86_64-linux-gnu
    )

    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB OR NOT TENSORRT_ONNXPARSER_LIB)
        message(FATAL_ERROR
            "TensorRT 를 못 찾음\n"
            "  include : ${TENSORRT_INCLUDE_DIR}\n"
            "  nvinfer : ${TENSORRT_LIB}\n"
            "  nvonnxparser : ${TENSORRT_ONNXPARSER_LIB}\n"
            "JetPack 이면 설치 상태를, x86 이면 -DTENSORRT_ROOT 를 확인할 것")
    endif()

    message(STATUS "TensorRT include : ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT nvinfer : ${TENSORRT_LIB}")
    message(STATUS "TensorRT parser  : ${TENSORRT_ONNXPARSER_LIB}")
    message(STATUS "CUDA toolkit     : ${CUDAToolkit_VERSION}")
endif()

# 각 모듈의 빌드 설정은 해당 디렉터리의 CMakeLists.txt에서 관리함
add_subdirectory(perception)
add_subdirectory(adas)
add_subdirectory(logging)
add_subdirectory(apps)