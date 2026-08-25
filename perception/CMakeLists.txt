# perception 모듈
# 객체 검출/추적 등 일반적인 인지 기능을 독립적인 라이브러리로 관리한다.

add_library(perception_core STATIC
    src/MultiObjectTracker.cpp
)

# perception의 공개 헤더 경로를 지정한다.
# perception_core를 사용하는 다른 타깃도 이 include 경로를 사용할 수 있다.
target_include_directories(perception_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# perception_core가 실제로 사용하는 OpenCV 모듈만 연결한다.
target_link_libraries(perception_core PUBLIC
    opencv_core
    opencv_video
)