cmake_minimum_required(VERSION 3.16)

project(blackbox_adas_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(adas
    src/main.cpp
)

target_include_directories(adas PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(adas PRIVATE
    ${OpenCV_LIBS}
)