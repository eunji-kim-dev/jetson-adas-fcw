add_library(adas_fcw STATIC
    src/LeadSelector.cpp
    src/BackgroundMotion.cpp
    src/RiskAnalyzer.cpp
    src/RoadHomography.cpp
    src/WarningPolicy.cpp
)

target_include_directories(adas_fcw PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# opencv_imgproc: LeadSelector.cpp의 pointPolygonTest, RoadHomography.cpp의 getPerspectiveTransform (헤더에는 노출 안 됨)
target_link_libraries(adas_fcw
    PUBLIC perception_core
    PRIVATE opencv_imgproc opencv_video
)
