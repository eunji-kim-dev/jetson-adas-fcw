# 실행 로그(raw_frame_log.csv / run_summary.json) 작성 라이브러리
# perception / adas 를 모르고, OpenCV 에도 의존하지 않는다. 앱만 링크한다.

set(BUILD_INFO_DIR ${CMAKE_BINARY_DIR}/generated/logging)
set(BUILD_INFO_HEADER ${BUILD_INFO_DIR}/BuildInfo.hpp)

# cmake --build 마다 실행되어 git commit / dirty 를 헤더로 생성
# (configure 시점에 넣으면 커밋 후 재빌드해도 옛 해시가 남는다)
add_custom_target(build_info ALL
    COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/BuildInfo.hpp.in
        -DOUTPUT=${BUILD_INFO_HEADER}
        -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
        "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GitInfo.cmake
    BYPRODUCTS ${BUILD_INFO_HEADER}
    COMMENT "BuildInfo.hpp 생성 (git commit / dirty)"
    VERBATIM
)

add_library(run_logger STATIC
    src/RunLogger.cpp
)
add_dependencies(run_logger build_info)

target_include_directories(run_logger
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${BUILD_INFO_DIR}
)
