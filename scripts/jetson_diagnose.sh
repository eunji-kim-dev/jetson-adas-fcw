#!/usr/bin/env bash

# Jetson 첫 세션 환경 진단 스크립트
# - 설치/설정 변경 없이 현재 상태만 확인
# - 한 항목이 실패해도 나머지 진단은 계속 진행
# - 화면 출력과 동시에 results/diagnostics/에 로그 저장

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/results/diagnostics"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/jetson_diagnose_${TIMESTAMP}.txt"

mkdir -p "${LOG_DIR}"


main() {
    echo "Log file: ${LOG_FILE}"
    echo

    echo "===== Jetson Diagnose ====="
    echo


    # ------------------------------------------------------------
    # 1. Jetson 보드 / JetPack / RAM
    # ------------------------------------------------------------

    echo "[1] Jetson Board / L4T / JetPack / RAM"

    echo "Board model:"
    if [ -r /proc/device-tree/model ]; then
        echo -n "  "
        tr -d '\0' < /proc/device-tree/model
        echo
    else
        echo "  NOT FOUND: /proc/device-tree/model"
    fi

    echo
    echo "L4T:"
    if [ -f /etc/nv_tegra_release ]; then
        cat /etc/nv_tegra_release
    else
        echo "  NOT FOUND: /etc/nv_tegra_release"
    fi

    echo
    echo "JetPack package:"
    if command -v dpkg-query >/dev/null 2>&1; then
        jetpack_version="$(
            dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null
        )"

        if [ -n "${jetpack_version}" ]; then
            echo "  ${jetpack_version}"
        else
            echo "  nvidia-jetpack package: NOT FOUND"
        fi
    else
        echo "  dpkg-query: NOT FOUND"
    fi

    echo
    echo "RAM:"
    if command -v free >/dev/null 2>&1; then
        free -h
    else
        echo "  free: NOT FOUND"
    fi

    echo


    # ------------------------------------------------------------
    # 2. CMake
    # ------------------------------------------------------------

    echo "[2] CMake"

    if command -v cmake >/dev/null 2>&1; then
        cmake --version | head -n 1
    else
        echo "NOT FOUND: cmake"
    fi

    echo


    # ------------------------------------------------------------
    # 3. C++ Compiler / C++17
    # ------------------------------------------------------------

    echo "[3] C++ Compiler"

    if command -v g++ >/dev/null 2>&1; then
        g++ --version | head -n 1

        if printf \
'#include <filesystem>\n#include <optional>\nint main(){std::optional<int> v=1; return (v.has_value() && std::filesystem::exists("/")) ? 0 : 1;}\n' \
            | g++ -std=c++17 -x c++ - \
                -o /tmp/jetson_cpp17_test >/dev/null 2>&1; then

            echo "C++17 filesystem/optional: OK"
            rm -f /tmp/jetson_cpp17_test
        else
            echo "C++17 filesystem/optional: FAIL"
            rm -f /tmp/jetson_cpp17_test
        fi
    else
        echo "NOT FOUND: g++"
    fi

    echo


    # ------------------------------------------------------------
    # 4. OpenCV
    # ------------------------------------------------------------

    echo "[4] OpenCV"

    if command -v opencv_version >/dev/null 2>&1; then
        echo -n "Version: "
        opencv_version
    else
        echo -n "Version (from soname): "

        version_from_soname="$(
            ldconfig -p 2>/dev/null \
                | grep -m1 "libopencv_core.so." \
                | sed 's/.*libopencv_core\.so\.//;s/ .*//'
        )"

        if [ -n "${version_from_soname}" ]; then
            echo "${version_from_soname}"
        else
            echo "UNKNOWN"
        fi
    fi

    echo
    echo "Required modules:"

    for module in core imgproc videoio dnn video; do
        if ldconfig -p 2>/dev/null \
            | grep -q "libopencv_${module}\.so"; then

            echo "  ${module}: OK"
        else
            echo "  ${module}: NOT FOUND"
        fi
    done

    echo


    # ------------------------------------------------------------
    # 5. OpenCV freetype
    # ------------------------------------------------------------

    echo "[5] OpenCV freetype"

    freetype_lib=0
    freetype_header=0
    freetype_cmake=0

    if ldconfig -p 2>/dev/null \
        | grep -q "libopencv_freetype"; then

        echo "library: FOUND"
        freetype_lib=1
    else
        echo "library: NOT FOUND"
    fi

    if [ -f /usr/include/opencv4/opencv2/freetype.hpp ] || \
       [ -f /usr/local/include/opencv4/opencv2/freetype.hpp ]; then

        echo "header: FOUND"
        freetype_header=1
    else
        echo "header: NOT FOUND"
    fi

    if command -v cmake >/dev/null 2>&1; then
        FREETYPE_PROBE_DIR="$(
            mktemp -d /tmp/jetson_opencv_freetype.XXXXXX 2>/dev/null
        )"

        if [ -n "${FREETYPE_PROBE_DIR}" ]; then

            cat > "${FREETYPE_PROBE_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(opencv_freetype_probe CXX)
find_package(OpenCV REQUIRED COMPONENTS freetype)
EOF

            if cmake \
                -S "${FREETYPE_PROBE_DIR}" \
                -B "${FREETYPE_PROBE_DIR}/build" \
                >/dev/null 2>&1; then

                echo "CMake component: FOUND"
                freetype_cmake=1
            else
                echo "CMake component: NOT FOUND"
            fi

            rm -rf "${FREETYPE_PROBE_DIR}"
        else
            echo "CMake component: UNKNOWN (temporary directory creation failed)"
        fi
    else
        echo "CMake component: UNKNOWN (cmake not found)"
    fi

    if [ "${freetype_lib}" -eq 1 ] && \
       [ "${freetype_header}" -eq 1 ] && \
       [ "${freetype_cmake}" -eq 1 ]; then

        echo "Build option: ENABLE_FREETYPE=ON 사용 가능"
    else
        echo "Build option: -DENABLE_FREETYPE=OFF 사용"
    fi

    echo


    # ------------------------------------------------------------
    # 6. Network / 모델 / 입력 영상
    # ------------------------------------------------------------

    echo "[6] Network / Input files"

    echo "Network:"

    if command -v curl >/dev/null 2>&1; then
        if curl -Is --max-time 5 \
            https://github.com >/dev/null 2>&1; then

            echo "  external network: AVAILABLE"
        else
            echo "  external network: UNAVAILABLE"
        fi
    else
        echo "  external network: UNKNOWN (curl not found)"
    fi

    echo
    echo "Expected FNV hashes:"
    echo "  input : fnv1a64:fe5e0610a5c0e53f"
    echo "  model : fnv1a64:9c5dac75fdcfb621"

    EXPECTED_MODEL_MD5="a933e257dfd691fcd8e0576013a43181"
    EXPECTED_INPUT_MD5="a09eecf3b4933273915fb6c75e23d221"

    MODEL_PATH="${PROJECT_ROOT}/models/yolov8n.onnx"
    INPUT_PATH="${PROJECT_ROOT}/videos/input.mp4"

    echo
    echo "Input files MD5:"

    if command -v md5sum >/dev/null 2>&1; then

        if [ -f "${MODEL_PATH}" ]; then
            actual_model_md5="$(
                md5sum "${MODEL_PATH}" | awk '{print $1}'
            )"

            echo "  model: ${actual_model_md5}"

            if [ "${actual_model_md5}" = "${EXPECTED_MODEL_MD5}" ]; then
                echo "    MD5: MATCH"
            else
                echo "    MD5: MISMATCH"
            fi
        else
            echo "  model: NOT FOUND"
            echo "    ${MODEL_PATH}"
        fi

        if [ -f "${INPUT_PATH}" ]; then
            actual_input_md5="$(
                md5sum "${INPUT_PATH}" | awk '{print $1}'
            )"

            echo "  input: ${actual_input_md5}"

            if [ "${actual_input_md5}" = "${EXPECTED_INPUT_MD5}" ]; then
                echo "    MD5: MATCH"
            else
                echo "    MD5: MISMATCH"
            fi
        else
            echo "  input: NOT FOUND"
            echo "    ${INPUT_PATH}"
        fi

    else
        echo "  md5sum: NOT FOUND"
    fi

    echo


    # ------------------------------------------------------------
    # 7. CUDA / TensorRT / 전력 도구
    # ------------------------------------------------------------

    echo "[7] CUDA / TensorRT / Jetson power tools"

    echo "CUDA:"

    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version | tail -n 1

    elif [ -f /usr/local/cuda/version.json ]; then
        if command -v python3 >/dev/null 2>&1; then
            python3 -c '
import json
with open("/usr/local/cuda/version.json") as f:
    data = json.load(f)
print("  CUDA", data["cuda"]["version"])
' 2>/dev/null \
                || echo "  CUDA version: UNKNOWN (version.json 파싱 실패)"
        else
            echo "  CUDA version: UNKNOWN (python3 not found)"
        fi

    elif [ -f /usr/local/cuda/version.txt ]; then
        cat /usr/local/cuda/version.txt

    else
        echo "  CUDA version: NOT FOUND"
    fi

    echo
    echo "TensorRT:"

    if command -v trtexec >/dev/null 2>&1; then
        trtexec --version 2>/dev/null \
            || echo "  trtexec: FOUND, version query failed"

    elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
        /usr/src/tensorrt/bin/trtexec --version 2>/dev/null \
            || echo "  trtexec: FOUND, version query failed"

    else
        echo "  trtexec: NOT FOUND"
    fi

    if command -v dpkg-query >/dev/null 2>&1; then
        tensorrt_packages="$(
            dpkg-query \
                -W \
                -f='${Package}: ${Version}\n' \
                'libnvinfer*' \
                2>/dev/null
        )"

        if [ -n "${tensorrt_packages}" ]; then
            echo "  Installed TensorRT packages:"
            echo "${tensorrt_packages}" | sed 's/^/    /'
        else
            echo "  libnvinfer packages: NOT FOUND"
        fi
    fi

    echo
    echo "Power tools:"

    if command -v nvpmodel >/dev/null 2>&1; then
        echo "nvpmodel: FOUND"

        echo "Current power mode:"
        if nvpmodel -q 2>/dev/null; then
            :
        else
            echo "  nvpmodel -q: 권한 없음 또는 조회 실패"
        fi
    else
        echo "nvpmodel: NOT FOUND"
    fi

    if command -v jetson_clocks >/dev/null 2>&1; then
        echo "jetson_clocks: FOUND"

        echo "jetson_clocks status:"
        if jetson_clocks --show 2>/dev/null; then
            :
        else
            echo "  --show: 권한 없음 또는 조회 실패"
        fi
    else
        echo "jetson_clocks: NOT FOUND"
    fi

    if command -v sudo >/dev/null 2>&1; then
        if sudo -n true >/dev/null 2>&1; then
            echo "sudo without password: AVAILABLE"
        else
            echo "sudo without password: NOT AVAILABLE"
            echo "  비밀번호 입력 sudo 가능 여부는 현장에서 sudo -v로 별도 확인"
        fi
    else
        echo "sudo: NOT FOUND"
    fi

    echo


    # ------------------------------------------------------------
    # 8. 카메라 / V4L2
    # ------------------------------------------------------------

    echo "[8] Video devices / Camera formats"

    video_devices=()

    while IFS= read -r device; do
        video_devices+=("${device}")
    done < <(compgen -G "/dev/video*" || true)

    if [ "${#video_devices[@]}" -eq 0 ]; then
        echo "Video devices: NONE"
    else
        echo "Video devices:"

        for device in "${video_devices[@]}"; do
            echo "  ${device}"
        done

        echo

        if command -v v4l2-ctl >/dev/null 2>&1; then

            for device in "${video_devices[@]}"; do
                echo "Formats for ${device}:"

                if v4l2-ctl \
                    --device="${device}" \
                    --list-formats-ext 2>/dev/null; then
                    :
                else
                    echo "  format query failed"
                fi

                echo
            done

        else
            echo "v4l2-ctl: NOT FOUND"
            echo "카메라 포맷은 현재 확인할 수 없음"
        fi
    fi

    echo


    # ------------------------------------------------------------
    # 9. 디스크 / 최종 체크
    # ------------------------------------------------------------

    echo "[9] Disk space / Checklist summary"

    echo "Project filesystem:"
    df -h "${PROJECT_ROOT}"

    echo
    echo "Checklist summary:"
    echo "  기본 포함       : 위 결과의 FOUND / OK / MATCH 항목"
    echo "  추가 확인 필요 : NOT FOUND / FAIL / MISMATCH 항목"

    echo
    echo "Build notes:"
    echo "  freetype 사용 불가:"
    echo "    -DENABLE_FREETYPE=OFF"

    echo
    echo "  한글 출력이 필요하면:"
    echo "    OpenCV freetype + 한글 폰트 별도 확인"

    echo
    echo "첫 Jetson 세션에서는 설치하지 않고 위 결과만 먼저 기록"

    echo
    echo "===== Diagnose Complete ====="
    echo "Saved to: ${LOG_FILE}"
}


main 2>&1 | tee "${LOG_FILE}"