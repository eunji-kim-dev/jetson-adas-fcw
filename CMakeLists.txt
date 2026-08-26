# 최상위 CMakeLists.txt
# 프로젝트 공통 설정만 정의하고,
# perception / adas / apps의 세부 빌드 설정은 각 디렉터리에 위임한다.

cmake_minimum_required(VERSION 3.16)
project(jetson_adas_fcw CXX)

# 프로젝트 전체에서 C++17을 사용한다.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 프로젝트에서 사용하는 OpenCV 모듈을 찾는다.
# freetype은 ADAS 앱의 한글 텍스트 렌더링에 사용한다.
# 실제로 각 타깃이 사용할 OpenCV 모듈은
# 하위 CMakeLists.txt의 target_link_libraries()에서 결정한다.
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    videoio
    dnn
    video
    freetype
)

# 각 모듈의 빌드 설정은 해당 디렉터리의 CMakeLists.txt에서 관리한다.
add_subdirectory(perception)
add_subdirectory(adas)
add_subdirectory(logging)
add_subdirectory(apps)