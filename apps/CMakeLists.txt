# common/: 두 앱이 공유하는 실행 옵션 파싱 (헤더 전용)
add_executable(adas fcw_demo/main.cpp)
target_include_directories(adas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(adas PRIVATE adas_fcw run_logger ${OpenCV_LIBS})

add_executable(perception_demo perception_demo/main.cpp)
target_include_directories(perception_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(perception_demo PRIVATE perception_core run_logger opencv_videoio opencv_imgproc)

# 카메라 입력 경로만 재는 도구 (추론 없음). Frame Age 하한·Frame Drop 확인용
add_executable(camera_probe camera_probe/main.cpp)
target_link_libraries(camera_probe PRIVATE perception_core)

if(ENABLE_FREETYPE)
    target_link_libraries(adas PRIVATE opencv_freetype)
    target_compile_definitions(adas PRIVATE ADAS_HAS_FREETYPE)
endif()
