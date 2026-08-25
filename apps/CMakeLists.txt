add_executable(adas fcw_demo/main.cpp)
target_link_libraries(adas PRIVATE adas_fcw ${OpenCV_LIBS})

add_executable(perception_demo perception_demo/main.cpp)
target_link_libraries(perception_demo PRIVATE perception_core opencv_videoio opencv_imgproc)
