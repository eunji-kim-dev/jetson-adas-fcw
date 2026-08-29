# 최상위 CMakeLists.txt
# 프로젝트 공통 설정만 정의하고,
# perception / adas / apps의 세부 빌드 설정은 각 디렉터리에 위임한다.

cmake_minimum_required(VERSION 3.16)
project(jetson_adas_fcw CXX)

# 프로젝트 전체에서 C++17을 사용한다.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 한글 오버레이에 사용하는 OpenCV freetype 모듈을 선택적으로 사용한다.
# 기본값은 ON이며, freetype이 없는 환경에서는
# -DENABLE_FREETYPE=OFF로 빌드할 수 있다.
option(ENABLE_FREETYPE "opencv_freetype 기반 한글 오버레이 사용" ON)

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

# 각 모듈의 빌드 설정은 해당 디렉터리의 CMakeLists.txt에서 관리한다.
add_subdirectory(perception)
add_subdirectory(adas)
add_subdirectory(logging)
add_subdirectory(apps)