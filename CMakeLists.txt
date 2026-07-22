cmake_minimum_required(VERSION 3.16)

project(blackbox_adas_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 한국어 표시를 위해 opencv_contrib의 freetype 모듈을 사용
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    videoio
    dnn
    video
    freetype
)

add_executable(adas
    src/main.cpp
    src/MultiObjectTracker.cpp
    src/RiskAnalyzer.cpp
)

target_include_directories(adas PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(adas PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(adas PRIVATE
    -Wall
    -Wextra
)
