#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../results/diagnostics"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/jetson_diagnose_${TIMESTAMP}.txt"

mkdir -p "${LOG_DIR}"

main() {
    echo "Log file: ${LOG_FILE}"
    echo

    echo "===== Jetson Diagnose ====="
    echo

    echo "[1] L4T / JetPack"
    if [ -f /etc/nv_tegra_release ]; then
        cat /etc/nv_tegra_release
    else
        echo "NOT FOUND: /etc/nv_tegra_release"
    fi

    echo

    echo "[2] CMake"
    if command -v cmake >/dev/null 2>&1; then
        cmake --version | head -n 1
    else
        echo "NOT FOUND: cmake"
    fi

    echo

    echo "[3] C++ Compiler"
    if command -v g++ >/dev/null 2>&1; then
        g++ --version | head -n 1
        if printf '#include <filesystem>\n#include <optional>\nint main(){return std::filesystem::exists("/") ? 0 : 1;}\n' \
            | g++ -std=c++17 -x c++ - -o /tmp/jetson_cpp17_test >/dev/null 2>&1; then
            echo "C++17 filesystem/optional: OK"
            rm -f /tmp/jetson_cpp17_test
        else
            echo "C++17 filesystem/optional: FAIL"
        fi
    else
        echo "NOT FOUND: g++"
    fi

    echo

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

    echo "Required modules:"
    for module in core imgproc videoio dnn video; do
        if ldconfig -p 2>/dev/null | grep -q "libopencv_${module}"; then
            echo "  ${module}: OK"
        else
            echo "  ${module}: NOT FOUND"
        fi
    done

    echo

    echo "[5] OpenCV freetype"

    freetype_lib=0
    freetype_header=0

    if ldconfig -p 2>/dev/null | grep -q "libopencv_freetype"; then
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

    if [ "${freetype_lib}" -eq 1 ] && [ "${freetype_header}" -eq 1 ]; then
        echo "Build option: ENABLE_FREETYPE=ON 사용 후보"
    else
        echo "Build option: -DENABLE_FREETYPE=OFF 사용"
    fi

    echo

    echo "[6] Network / Input files"

    echo "Network:"
    if command -v curl >/dev/null 2>&1; then
        if curl -Is --max-time 5 https://github.com >/dev/null 2>&1; then
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

    echo
    echo "Input files MD5:"

    EXPECTED_MODEL_MD5="a933e257dfd691fcd8e0576013a43181"
    EXPECTED_INPUT_MD5="a09eecf3b4933273915fb6c75e23d221"

    MODEL_PATH="${SCRIPT_DIR}/../models/yolov8n.onnx"
    INPUT_PATH="${SCRIPT_DIR}/../videos/input.mp4"

    if [ -f "${MODEL_PATH}" ]; then
        actual_model_md5="$(md5sum "${MODEL_PATH}" | awk '{print $1}')"
        echo "  model: ${actual_model_md5}"

        if [ "${actual_model_md5}" = "${EXPECTED_MODEL_MD5}" ]; then
            echo "    MD5: MATCH"
        else
            echo "    MD5: MISMATCH"
        fi
    else
        echo "  model: NOT FOUND"
    fi

    if [ -f "${INPUT_PATH}" ]; then
        actual_input_md5="$(md5sum "${INPUT_PATH}" | awk '{print $1}')"
        echo "  input: ${actual_input_md5}"

        if [ "${actual_input_md5}" = "${EXPECTED_INPUT_MD5}" ]; then
            echo "    MD5: MATCH"
        else
            echo "    MD5: MISMATCH"
        fi
    else
        echo "  input: NOT FOUND"
    fi

    echo

    echo "[7] Jetson power tools"

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
    else
        echo "jetson_clocks: NOT FOUND"
    fi

    if sudo -n true >/dev/null 2>&1; then
        echo "sudo without password: AVAILABLE"
    else
        echo "sudo without password: NOT AVAILABLE"
    fi

    echo

    echo "[8] Disk space"
    echo "Project filesystem:"
    df -h "${SCRIPT_DIR}/.."

    echo

    echo "[9] Checklist summary"
    echo "기본 포함       : 위 결과에서 FOUND / OK 항목 확인"
    echo "추가 설치 필요 : NOT FOUND / FAIL 항목 확인"
    echo
    echo "freetype이 없으면 -DENABLE_FREETYPE=OFF 사용"
    echo "한글 출력이 필요하면 freetype 및 한글 폰트 별도 확인"

    echo
    echo "===== Diagnose Complete ====="
    echo "Saved to: ${LOG_FILE}"
}

main 2>&1 | tee "${LOG_FILE}"