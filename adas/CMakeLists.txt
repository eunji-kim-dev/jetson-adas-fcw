# ADAS 모듈
# perception 결과를 입력으로 받아 FCW 위험 판단을 수행하는 라이브러리다.

add_library(adas_fcw STATIC
    src/RiskAnalyzer.cpp
)

# ADAS의 공개 헤더 경로를 지정한다.
# adas_fcw를 사용하는 다른 타깃에서도 이 헤더를 사용할 수 있다.
target_include_directories(adas_fcw PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ADAS는 perception 결과를 사용하므로 perception_core에 의존한다.
target_link_libraries(adas_fcw PUBLIC
    perception_core
)