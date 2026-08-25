# 실행 애플리케이션 모듈
# FCW 전체 실행 앱과 perception 독립 검증용 데모를 만든다.

add_executable(adas
    fcw_demo/main.cpp
)

# 실제 FCW 앱은 ADAS 로직과 OpenCV 전체 기능을 사용한다.
target_link_libraries(adas PRIVATE
    adas_fcw
    ${OpenCV_LIBS}
)

add_executable(perception_demo
    perception_demo/main.cpp
)

# perception_demo는 ADAS 없이 perception_core만 사용한다.
# 영상 입력/표시를 위해 videoio, highgui만 추가로 연결한다.
target_link_libraries(perception_demo PRIVATE
    perception_core
    opencv_videoio
    opencv_highgui
)