# 빌드 시점 git 정보 수집 → BuildInfo.hpp 생성
# 인자: SOURCE_DIR, TEMPLATE, OUTPUT, BUILD_TYPE, COMPILER
#
# dirty 판정: 추적 중인 파일의 미커밋 변경 (untracked 파일은 제외)
# dirty 상태에서 나온 측정은 무효로 본다 (8월 계획서 규칙)

find_package(Git QUIET)

set(GIT_COMMIT "unknown")
set(GIT_DIRTY 1)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE commit_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE commit_result
        ERROR_QUIET
    )
    if(commit_result EQUAL 0)
        set(GIT_COMMIT "${commit_output}")
    endif()

    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${SOURCE_DIR}" status --porcelain --untracked-files=no
        OUTPUT_VARIABLE status_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE status_result
        ERROR_QUIET
    )
    if(status_result EQUAL 0)
        if(status_output STREQUAL "")
            set(GIT_DIRTY 0)
        else()
            set(GIT_DIRTY 1)
        endif()
    endif()
endif()

configure_file("${TEMPLATE}" "${OUTPUT}.tmp" @ONLY)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUTPUT}.tmp" "${OUTPUT}")
file(REMOVE "${OUTPUT}.tmp")
