#!/usr/bin/env bash

# 고정된 측정 조건으로 ADAS 를 반복 실행하고 결과를 검증하는 벤치마크 하네스
#
# 사용법:
#   bash scripts/benchmark_harness.sh <실험이름> [옵션]
#   예: bash scripts/benchmark_harness.sh baseline_x86
#       bash scripts/benchmark_harness.sh baseline_jetson_cpu --clocks on --video off
#
# 옵션
#   --clocks on|off|asis   jetson_clocks 를 켜고/되돌리고/건드리지 않고 측정할지 (기본 asis)
#   --video  on|off|asis   결과 영상을 쓸지 (기본 asis — 바이너리 기본 동작 그대로)
#
# 사람이 입력하는 것은 실험 이름과 위 두 옵션뿐이고,
# 나머지 측정 조건은 아래 상수로 고정함
#
# Power Mode 는 nvpmodel 이 있으면 장비에서 읽고, 없으면 DEFAULT_POWER_MODE 를 씀
# Run 전후로 장비 상태(Power Mode, 클럭, 온도, 부하)를 device_before / device_after 에 남김
# Run 도중에는 tegrastats 를 같이 돌려서 온도와 전력을 시간축으로 남김

set -euo pipefail


# ------------------------------------------------------------
# 고정 측정 조건
# ------------------------------------------------------------

readonly RUN_COUNT=5
readonly WAIT_SECONDS=900
readonly WARMUP_FRAMES=15
readonly MEASURED_FRAMES=1237

readonly BACKEND="opencv_dnn"
readonly DEFAULT_POWER_MODE="ac_balanced"

readonly BINARY_REL="build/apps/adas"
readonly INPUT_REL="videos/input.mp4"
readonly MODEL_REL="models/yolov8n.onnx"

readonly GOLDEN_REL="results/golden_baseline.csv"
readonly GOLDEN_MD5="a0006c4a16dbe3f69c178fbc5c1b6b8e"

readonly SESSIONS_ROOT_REL="results/benchmark_sessions"
readonly ARCHIVES_ROOT_REL="results/benchmark_archives"
readonly RUNS_ROOT_REL="results/runs"


# ------------------------------------------------------------
# 측정 환경 조건
# ------------------------------------------------------------
# 여기 값들은 최적화 실험을 시작하기 전에 고정해 두는 것임
# 나중에 고칠 일이 생기면 작업일지에 이유를 남겨야 비교가 깨지지 않음

# Debug 빌드로 재면 Release 대비 몇 배가 느린데 숫자만 보면 구분이 안 됨
readonly REQUIRED_BUILD_TYPE="Release"

# 15 FPS 기준 1프레임 예산. 실제 계산은 tools/deadline_report.py 가 함
readonly DEADLINE_MS=66.7
# 지금은 처리시간 기준. V4L2 카메라를 붙이면 frame_age_ms 로 바꿀 자리임
readonly DEADLINE_TARGET="total_processing_ms"

readonly TEGRASTATS_INTERVAL_MS=1000

# 배경 작업이 도는 중에 시작하면 앞 회차와 조건이 달라짐
readonly IDLE_LOADAVG_MAX=0.60
readonly IDLE_WAIT_MAX_S=300

# 결과를 SD카드에 쓰면 I/O 대기가 프레임 시간에 섞임
# nvme 를 강제하려면 RESULT_DISK_REQUIRE=nvme 로 실행함
readonly RESULT_DISK_REQUIRE="${RESULT_DISK_REQUIRE:-}"

readonly TEGRASTATS_PARSER_REL="tools/parse_tegrastats.py"


# ------------------------------------------------------------
# 실행 위치 고정
# ------------------------------------------------------------
# adas 는 models/, results/ 를 상대 경로로 쓰므로
# 어디서 호출하든 프로젝트 루트에서 실행해야 함

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SCRIPT_DIR PROJECT_ROOT

cd "${PROJECT_ROOT}"


# ------------------------------------------------------------
# 인자 확인
# ------------------------------------------------------------

usage() {
    echo "사용법: bash scripts/benchmark_harness.sh <실험이름> [--clocks on|off|asis] [--video on|off|asis]" >&2
    echo "  예:   bash scripts/benchmark_harness.sh baseline_x86" >&2
    echo "  예:   bash scripts/benchmark_harness.sh baseline_jetson_cpu --clocks on --video off" >&2
    exit 2
}

EXPERIMENT_NAME=""
CLOCKS_MODE="asis"
VIDEO_MODE="asis"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clocks)
            [[ $# -ge 2 ]] || { echo "[ERROR] --clocks 에 값이 없음" >&2; usage; }
            CLOCKS_MODE="$2"
            shift 2
            ;;
        --video)
            [[ $# -ge 2 ]] || { echo "[ERROR] --video 에 값이 없음" >&2; usage; }
            VIDEO_MODE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "[ERROR] 모르는 옵션: $1" >&2
            usage
            ;;
        *)
            if [[ -n "${EXPERIMENT_NAME}" ]]; then
                echo "[ERROR] 실험 이름이 두 개임: ${EXPERIMENT_NAME}, $1" >&2
                usage
            fi
            EXPERIMENT_NAME="$1"
            shift
            ;;
    esac
done

if [[ -z "${EXPERIMENT_NAME}" ]]; then
    echo "[ERROR] 실험 이름이 없음" >&2
    usage
fi

# 실험 이름은 그대로 디렉터리 이름이 되므로 공백과 경로 문자를 막음
if [[ ! "${EXPERIMENT_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] 실험 이름에는 영문/숫자/밑줄/하이픈만 사용: ${EXPERIMENT_NAME}" >&2
    exit 2
fi

if [[ ! "${CLOCKS_MODE}" =~ ^(on|off|asis)$ ]]; then
    echo "[ERROR] --clocks 값이 이상함: ${CLOCKS_MODE}" >&2
    usage
fi

if [[ ! "${VIDEO_MODE}" =~ ^(on|off|asis)$ ]]; then
    echo "[ERROR] --video 값이 이상함: ${VIDEO_MODE}" >&2
    usage
fi

readonly EXPERIMENT_NAME CLOCKS_MODE VIDEO_MODE


# ------------------------------------------------------------
# 입력 영상 이름에서 FCW 결과 CSV 경로를 유도
# ------------------------------------------------------------
# main.cpp: results/<입력파일 stem>_frames.csv
# videos/input.mp4 -> results/input_frames.csv

input_basename="$(basename "${INPUT_REL}")"
readonly INPUT_STEM="${input_basename%.*}"
readonly FCW_CSV_REL="results/${INPUT_STEM}_frames.csv"


# ------------------------------------------------------------
# Preflight 보고용 도우미
# ------------------------------------------------------------

preflight_failures=0

ok() {
    echo "  [OK]   $*"
}

warn() {
    echo "  [WARN] $*"
}

fail() {
    echo "  [FAIL] $*" >&2
    preflight_failures=$((preflight_failures + 1))
}

require_command() {
    local name="$1"

    if command -v "${name}" >/dev/null 2>&1; then
        ok "명령 사용 가능: ${name}"
    else
        fail "명령 없음: ${name}"
    fi
}

require_file() {
    local path="$1"
    local label="$2"

    if [[ -f "${path}" ]]; then
        ok "${label}: ${path}"
    else
        fail "${label} 없음: ${path}"
    fi
}


# ------------------------------------------------------------
# sudo 유지
# ------------------------------------------------------------
# tegrastats 와 jetson_clocks 는 권한이 필요한데
# sudo 기본 타임스탬프가 15분이라 5회 세션(약 2시간) 중간에 만료됨
# 시작할 때 한 번 인증하고 백그라운드에서 갱신함

SUDO_OK=0
SUDO_KEEPALIVE_PID=""

ensure_sudo() {
    if ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi

    # 이미 통과 상태면 물어보지 않고 넘어감
    if sudo -n true 2>/dev/null; then
        SUDO_OK=1
    elif sudo -v 2>/dev/null; then
        SUDO_OK=1
    else
        return 1
    fi

    (
        while true; do
            sudo -n true 2>/dev/null || exit 0
            sleep 60
            kill -0 "$$" 2>/dev/null || exit 0   # 부모가 죽으면 같이 끝냄
        done
    ) &
    SUDO_KEEPALIVE_PID=$!
    return 0
}

# 권한이 있으면 sudo 로, 없으면 그냥 실행함
priv_run() {
    if [[ "${SUDO_OK}" -eq 1 ]]; then
        sudo -n "$@"
    else
        "$@"
    fi
}


# ------------------------------------------------------------
# 빌드 설정 읽기
# ------------------------------------------------------------
# C++ 를 건드리지 않고 CMakeCache.txt 에서 읽음

read_cache_value() {
    local key="$1"
    local cache="build/CMakeCache.txt"

    if [[ ! -f "${cache}" ]]; then
        echo ""
        return 0
    fi
    grep -m1 "^${key}:" "${cache}" 2>/dev/null | cut -d= -f2- || true
}


# ------------------------------------------------------------
# 결과 저장 디스크 판별
# ------------------------------------------------------------

detect_result_disk() {
    df -P results 2>/dev/null | awk 'NR==2 {print $1}'
}

classify_disk() {
    case "$1" in
        *mmcblk*) echo "sdcard" ;;
        *nvme*)   echo "nvme" ;;
        "")       echo "unknown" ;;
        *)        echo "other" ;;
    esac
}


# ------------------------------------------------------------
# jetson_clocks
# ------------------------------------------------------------

clocks_available() {
    command -v jetson_clocks >/dev/null 2>&1
}

snapshot_clocks() {
    local out_file="$1"

    if clocks_available; then
        if ! priv_run jetson_clocks --show > "${out_file}" 2>&1; then
            echo "READ FAILED" > "${out_file}"
        fi
    else
        echo "NOT FOUND" > "${out_file}"
    fi
}

apply_clocks() {
    if ! clocks_available; then
        if [[ "${CLOCKS_MODE}" != "asis" ]]; then
            warn "jetson_clocks 없음 — --clocks ${CLOCKS_MODE} 를 적용하지 못함"
        fi
        return 0
    fi

    case "${CLOCKS_MODE}" in
        on)
            if priv_run jetson_clocks >/dev/null 2>&1; then
                ok "jetson_clocks 적용함 (클럭 최대 고정)"
                return 0
            fi
            fail "jetson_clocks 적용 실패 — sudo 권한 확인 필요"
            return 1
            ;;
        off)
            if priv_run jetson_clocks --restore >/dev/null 2>&1; then
                ok "jetson_clocks 되돌림 (DVFS 원래대로)"
                return 0
            fi
            fail "jetson_clocks 복원 실패"
            return 1
            ;;
        asis)
            ok "jetson_clocks 건드리지 않음"
            return 0
            ;;
    esac
}


# ------------------------------------------------------------
# tegrastats 동시 기록
# ------------------------------------------------------------
# Run 전후 스냅샷만으로는 구간 중간에 온도가 어떻게 올라갔는지 못 봄
# Run 이 도는 동안 1초마다 찍어서 프레임 시간과 같은 시간축에 올림

TEGRASTATS_PID=""

tegrastats_available() {
    command -v tegrastats >/dev/null 2>&1
}

tegrastats_stop_cmd() {
    priv_run tegrastats --stop
}

start_tegrastats() {
    local out_file="$1"

    TEGRASTATS_PID=""

    if ! tegrastats_available; then
        echo "NOT FOUND" > "${out_file}.skipped"
        return 0
    fi

    # 이미 떠 있는 인스턴스가 있으면 새로 안 뜨고 조용히 실패함
    tegrastats_stop_cmd >/dev/null 2>&1 || true

    priv_run tegrastats \
        --interval "${TEGRASTATS_INTERVAL_MS}" \
        --logfile "${out_file}" &
    TEGRASTATS_PID=$!

    # 첫 줄이 찍힐 때까지 기다림
    sleep 2

    if [[ ! -s "${out_file}" ]]; then
        printf '  [WARN] tegrastats 로그가 비어 있음 — 권한이나 중복 실행 확인 필요\n' >&2
    fi
}

stop_tegrastats() {
    if [[ -z "${TEGRASTATS_PID}" ]]; then
        return 0
    fi

    tegrastats_stop_cmd >/dev/null 2>&1 || true
    wait "${TEGRASTATS_PID}" 2>/dev/null || true
    TEGRASTATS_PID=""
}


# ------------------------------------------------------------
# 종료 정리
# ------------------------------------------------------------
# Ctrl+C 로 끊어도 tegrastats 가 남지 않게 함
# 남아 있으면 다음 세션에서 tegrastats 가 아예 안 뜸

cleanup_on_exit() {
    stop_tegrastats

    if [[ -n "${SUDO_KEEPALIVE_PID}" ]]; then
        kill "${SUDO_KEEPALIVE_PID}" 2>/dev/null || true
        SUDO_KEEPALIVE_PID=""
    fi
}

# INT / TERM 을 EXIT 과 같은 트랩에 묶으면
# 정리만 하고 스크립트가 계속 돌아감. 여기서 명시적으로 끝내야 함
on_interrupt() {
    printf '\n[ABORT] 중단 신호를 받음 — 정리하고 종료함\n' >&2
    exit 130
}

trap cleanup_on_exit EXIT
trap on_interrupt INT TERM


# ------------------------------------------------------------
# 유휴 대기
# ------------------------------------------------------------
# 배경 작업이 도는 중에 시작하면 그 Run 만 느리게 나옴
# 내려갈 때까지 기다리고, 안 내려가면 그대로 가되 의심값으로 표시함

loadavg_1m() {
    awk '{print $1}' /proc/loadavg 2>/dev/null || echo "0"
}

wait_for_idle() {
    local waited=0
    local la

    while (( waited < IDLE_WAIT_MAX_S )); do
        la="$(loadavg_1m)"

        if awk -v a="${la}" -v b="${IDLE_LOADAVG_MAX}" 'BEGIN { exit !(a < b) }'; then
            printf '  loadavg : %s (기준 %s 미만) — 시작함\n' "${la}" "${IDLE_LOADAVG_MAX}"
            return 0
        fi

        printf '  loadavg : %s 가 기준 %s 보다 높음 — %d s 대기 중\n' \
            "${la}" "${IDLE_LOADAVG_MAX}" "${waited}"
        sleep 5
        waited=$((waited + 5))
    done

    printf '  [WARN] loadavg 가 %d s 동안 안 내려감 — 진행하되 의심값으로 표시함\n' \
        "${IDLE_WAIT_MAX_S}" >&2
    return 1
}


# ------------------------------------------------------------
# 장비 상태 기록
# ------------------------------------------------------------
# Jetson 이 아니거나 도구가 없어도 멈추지 않고 NOT FOUND / N/A 로 남김
# 온도는 thermal_zone 의 temp 가 milli-degree 라 1000 으로 나눠서 같이 적음

record_device_state() {
    local out_file="$1"
    local label="$2"

    {
        printf 'label     : %s\n' "${label}"
        printf 'timestamp : %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        echo

        echo "[nvpmodel]"
        if command -v nvpmodel >/dev/null 2>&1; then
            nvpmodel -q 2>/dev/null || echo "query failed"
        else
            echo "NOT FOUND"
        fi
        echo

        echo "[jetson_clocks]"
        if command -v jetson_clocks >/dev/null 2>&1; then
            if jetson_clocks --show 2>/dev/null; then
                :
            elif sudo -n jetson_clocks --show 2>/dev/null; then
                :
            else
                echo "권한 없음 (sudo 필요)"
            fi
        else
            echo "NOT FOUND"
        fi
        echo

        echo "[thermal]"
        local zone zone_type temp_raw found_zone=0
        for zone in /sys/devices/virtual/thermal/thermal_zone*; do
            [[ -r "${zone}/temp" ]] || continue
            found_zone=1
            zone_type="$(cat "${zone}/type" 2>/dev/null || echo unknown)"
            temp_raw="$(cat "${zone}/temp" 2>/dev/null || echo "")"

            if [[ "${temp_raw}" =~ ^-?[0-9]+$ ]]; then
                printf '%s: %s (%s C)\n' "${zone_type}" "${temp_raw}" \
                    "$(awk -v t="${temp_raw}" 'BEGIN { printf "%.1f", t / 1000 }')"
            else
                printf '%s: read failed\n' "${zone_type}"
            fi
        done
        [[ "${found_zone}" -eq 1 ]] || echo "N/A"
        echo

        echo "[cpufreq kHz]"
        local cpu found_cpu=0
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
            [[ -r "${cpu}/cpufreq/scaling_cur_freq" ]] || continue
            found_cpu=1
            printf '%s: %s\n' "$(basename "${cpu}")" \
                "$(cat "${cpu}/cpufreq/scaling_cur_freq" 2>/dev/null || echo '?')"
        done
        [[ "${found_cpu}" -eq 1 ]] || echo "N/A"
        echo

        echo "[loadavg]"
        cat /proc/loadavg 2>/dev/null || echo "N/A"
        echo

        echo "[memory]"
        free -m 2>/dev/null || echo "N/A"
    } > "${out_file}"
}


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

echo "===== Preflight ====="
echo "  project root: ${PROJECT_ROOT}"
echo

echo "필요한 명령"
require_command md5sum
require_command tar
require_command date
require_command awk

if command -v python3 >/dev/null 2>&1; then
    ok "명령 사용 가능: python3"
else
    warn "python3 없음 — tegrastats 로그 요약은 생략됨"
fi
echo

echo "측정 대상 파일"
if [[ -x "${BINARY_REL}" ]]; then
    ok "실행 파일: ${BINARY_REL}"
else
    fail "실행 파일이 없거나 실행 권한 없음: ${BINARY_REL}"
fi
require_file "${INPUT_REL}" "입력 영상"
require_file "${MODEL_REL}" "모델"
echo

echo "빌드 설정"
# Debug 로 재면 Release 대비 몇 배 느린데 결과 숫자만 봐서는 구분이 안 됨
# 기록만 하지 않고 여기서 막음
BUILD_TYPE="$(read_cache_value 'CMAKE_BUILD_TYPE')"
CXX_FLAGS_RELEASE="$(read_cache_value 'CMAKE_CXX_FLAGS_RELEASE')"
[[ -n "${BUILD_TYPE}" ]] || BUILD_TYPE="UNKNOWN"
readonly BUILD_TYPE CXX_FLAGS_RELEASE

if [[ "${BUILD_TYPE}" == "${REQUIRED_BUILD_TYPE}" ]]; then
    ok "build type: ${BUILD_TYPE}"
    ok "cxx flags (release): ${CXX_FLAGS_RELEASE:-(빈 값)}"
else
    fail "build type 이 ${BUILD_TYPE} 임 — ${REQUIRED_BUILD_TYPE} 이 아니면 측정하지 않음"
    fail "  cmake -S . -B build -DCMAKE_BUILD_TYPE=${REQUIRED_BUILD_TYPE} && cmake --build build -j4"
fi

# 설정만 바꾸고 재빌드를 안 한 상태를 잡음
if [[ -f "build/CMakeCache.txt" && -f "${BINARY_REL}" ]]; then
    if [[ "build/CMakeCache.txt" -nt "${BINARY_REL}" ]]; then
        fail "CMakeCache.txt 가 바이너리보다 새로움 — 재빌드 필요"
    else
        ok "바이너리가 CMakeCache 보다 최신"
    fi
fi
echo

echo "결과 저장 위치"
# SD카드에 결과를 쓰면 I/O 대기가 프레임 시간에 섞여 들어감
RESULT_DISK="$(detect_result_disk)"
RESULT_DISK_KIND="$(classify_disk "${RESULT_DISK}")"
readonly RESULT_DISK RESULT_DISK_KIND

ok "results 디스크: ${RESULT_DISK:-unknown} (${RESULT_DISK_KIND})"

if [[ -n "${RESULT_DISK_REQUIRE}" ]]; then
    if [[ "${RESULT_DISK_KIND}" == "${RESULT_DISK_REQUIRE}" ]]; then
        ok "요구 디스크 종류와 일치: ${RESULT_DISK_REQUIRE}"
    else
        fail "요구 디스크는 ${RESULT_DISK_REQUIRE} 인데 실제는 ${RESULT_DISK_KIND} 임"
    fi
elif [[ "${RESULT_DISK_KIND}" == "sdcard" ]]; then
    warn "결과가 SD카드에 쌓임 — 프레임 시간에 I/O 대기가 섞일 수 있음"
    warn "NVMe 로 옮기려면 프로젝트를 /mnt/ssd 아래로 옮기고 다시 실행"
fi
echo

echo "결과 영상"
# --video off 를 주려면 바이너리가 그 플래그를 받아야 함
VIDEO_FLAG_SUPPORTED=0
if [[ -x "${BINARY_REL}" ]]; then
    # --help 는 바이너리가 종료 코드 1 로 끝내므로 pipefail 에 걸리지 않게 먼저 받아둠
    help_text="$("./${BINARY_REL}" --help 2>&1 || true)"
    if grep -q -- '--no-video' <<< "${help_text}"; then
        VIDEO_FLAG_SUPPORTED=1
    fi
fi
readonly VIDEO_FLAG_SUPPORTED

case "${VIDEO_MODE}" in
    off)
        if [[ "${VIDEO_FLAG_SUPPORTED}" -eq 1 ]]; then
            ok "결과 영상 끄고 측정함 (--no-video)"
        else
            fail "바이너리에 --no-video 플래그가 없음 — --video off 를 쓸 수 없음"
            fail "  './${BINARY_REL} --help' 로 실제 플래그 이름을 확인할 것"
        fi
        ;;
    on)
        warn "결과 영상을 쓰면서 측정함 — 쓰기 시간이 프레임 시간에 포함될 수 있음"
        ;;
    asis)
        warn "결과 영상 여부를 지정하지 않음 (바이너리 기본 동작)"
        ;;
esac
echo

echo "Golden 기준"
if [[ -f "${GOLDEN_REL}" ]]; then
    golden_actual_md5="$(md5sum "${GOLDEN_REL}" | awk '{print $1}')"

    if [[ "${golden_actual_md5}" == "${GOLDEN_MD5}" ]]; then
        ok "golden MD5 일치: ${golden_actual_md5}"
    else
        fail "golden MD5 불일치"
        fail "  기대: ${GOLDEN_MD5}"
        fail "  실제: ${golden_actual_md5}"
    fi
else
    fail "golden CSV 없음: ${GOLDEN_REL}"
fi
echo

echo "Power Mode"
# nvpmodel 이 있으면 장비가 말하는 값을 쓰고, 없으면 기본값을 씀
# 사람이 값을 치지 않게 해서 로그에 틀린 Power Mode 가 남는 일을 막음
POWER_MODE=""
POWER_MODE_SOURCE=""

if command -v nvpmodel >/dev/null 2>&1; then
    POWER_MODE="$(
        nvpmodel -q 2>/dev/null \
            | tr -d '\r' \
            | sed -n 's/^NV Power Mode: *//p' \
            | head -n 1 \
            || true
    )"

    if [[ -n "${POWER_MODE}" ]]; then
        POWER_MODE_SOURCE="nvpmodel"
    else
        warn "nvpmodel 은 있으나 Power Mode 를 읽지 못함 — 기본값 사용"
    fi
fi

if [[ -z "${POWER_MODE}" ]]; then
    POWER_MODE="${DEFAULT_POWER_MODE}"
    POWER_MODE_SOURCE="default"
fi

readonly POWER_MODE POWER_MODE_SOURCE
ok "power mode: ${POWER_MODE} (${POWER_MODE_SOURCE})"
echo

echo "권한"
if tegrastats_available || clocks_available; then
    if ensure_sudo; then
        ok "sudo 사용 가능 — 세션 동안 갱신함"
    else
        warn "sudo 를 못 씀 — tegrastats / jetson_clocks 가 제한될 수 있음"
    fi
else
    ok "sudo 필요 없음 (tegrastats, jetson_clocks 둘 다 없음)"
fi
echo

echo "클럭 상태"
if ! apply_clocks; then
    fail "클럭을 요청한 상태(${CLOCKS_MODE})로 맞추지 못함 — 조건이 달라지므로 중단"
fi
echo

echo "온도 기록"
if tegrastats_available; then
    ok "tegrastats 사용 가능 — ${TEGRASTATS_INTERVAL_MS} ms 간격으로 기록함"

    if [[ -f "${TEGRASTATS_PARSER_REL}" ]]; then
        ok "파서: ${TEGRASTATS_PARSER_REL}"
    else
        warn "파서 없음: ${TEGRASTATS_PARSER_REL} — 로그만 남기고 요약은 생략"
    fi
else
    warn "tegrastats 없음 (x86 이면 정상) — 온도/전력 기록 생략"
fi
echo

echo "Git 상태 (측정 중단 사유는 아님)"
if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    git_head="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"

    if [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then
        ok "추적 파일 미커밋 변경 없음 (HEAD ${git_head})"
    else
        warn "추적 파일에 미커밋 변경 있음 (HEAD ${git_head})"
        warn "실측 전에는 커밋 후 재빌드 필요 — 지금은 그대로 진행"
    fi
else
    warn "git 저장소가 아니거나 git 없음 — 커밋 상태 확인 생략"
fi
echo


# ------------------------------------------------------------
# Session ID / Run ID
# ------------------------------------------------------------

SESSION_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
readonly SESSION_TIMESTAMP
readonly SESSION_ID="${EXPERIMENT_NAME}_${SESSION_TIMESTAMP}"
readonly SESSION_DIR="${SESSIONS_ROOT_REL}/${SESSION_ID}"

run_id_for() {
    printf '%s_r%02d' "${SESSION_ID}" "$1"
}

echo "세션 디렉터리"
if [[ -e "${SESSION_DIR}" ]]; then
    fail "세션 디렉터리가 이미 있음: ${SESSION_DIR}"
else
    ok "새 세션 경로: ${SESSION_DIR}"
fi
echo


# ------------------------------------------------------------
# Preflight 판정
# ------------------------------------------------------------

if [[ "${preflight_failures}" -ne 0 ]]; then
    echo "[ABORT] Preflight 실패 ${preflight_failures}건 — 측정을 시작하지 않음" >&2
    exit 1
fi

echo "[PASS] Preflight 통과"
echo


# ------------------------------------------------------------
# 측정 계획 출력
# ------------------------------------------------------------

mkdir -p "${SESSION_DIR}"
mkdir -p "${ARCHIVES_ROOT_REL}"

echo "===== 측정 계획 ====="
printf '  session id      : %s\n' "${SESSION_ID}"
printf '  session dir     : %s\n' "${SESSION_DIR}"
printf '  binary          : %s\n' "${BINARY_REL}"
printf '  input           : %s\n' "${INPUT_REL}"
printf '  model           : %s\n' "${MODEL_REL}"
printf '  backend         : %s\n' "${BACKEND}"
printf '  build type      : %s\n' "${BUILD_TYPE}"
printf '  power mode      : %s (%s)\n' "${POWER_MODE}" "${POWER_MODE_SOURCE}"
printf '  clocks mode     : %s\n' "${CLOCKS_MODE}"
printf '  video mode      : %s\n' "${VIDEO_MODE}"
printf '  result disk     : %s (%s)\n' "${RESULT_DISK:-unknown}" "${RESULT_DISK_KIND}"
printf '  warmup frames   : %s\n' "${WARMUP_FRAMES}"
printf '  measured frames : %s\n' "${MEASURED_FRAMES}"
printf '  total frames    : %s\n' "$((WARMUP_FRAMES + MEASURED_FRAMES))"
printf '  run count       : %s\n' "${RUN_COUNT}"
printf '  wait between    : %s s\n' "${WAIT_SECONDS}"
printf '  idle gate       : loadavg < %s, 최대 %s s 대기\n' "${IDLE_LOADAVG_MAX}" "${IDLE_WAIT_MAX_S}"
printf '  deadline        : %s ms (대상 %s)\n' "${DEADLINE_MS}" "${DEADLINE_TARGET}"
printf '  fcw csv         : %s\n' "${FCW_CSV_REL}"
printf '  run log root    : %s\n' "${RUNS_ROOT_REL}"
echo

echo "  예정 Run ID"
for ((run_index = 1; run_index <= RUN_COUNT; run_index++)); do
    printf '    run %02d : %s\n' "${run_index}" "$(run_id_for "${run_index}")"
done
echo


# ------------------------------------------------------------
# Session Summary 초기화
# ------------------------------------------------------------
# Run 이 끝날 때마다 한 줄씩 추가함
# 세션이 중간에 끊겨도 그때까지의 기록은 남음
#
# result 는 정확성 판정(golden MD5 등), quality 는 측정 조건 판정으로 나눔
# 온도 기록이 빠진 Run 은 결과가 맞아도 나중에 해석이 안 되므로 따로 표시함

readonly SESSION_SUMMARY="${SESSION_DIR}/session_summary.csv"

printf 'run_index,run_id,started_at,elapsed_s,exit_code,fcw_csv_exists,fcw_csv_md5,md5_match,raw_frame_log_exists,run_summary_exists,result,quality,quality_reasons,tegrastats_lines,clocks_changed,idle_ok\n' \
    > "${SESSION_SUMMARY}"


# ------------------------------------------------------------
# Run 1회 실행과 판정
# ------------------------------------------------------------
# PASS 조건 (전부 만족해야 함)
#   1. ADAS 종료 코드 0
#   2. FCW 결과 CSV 존재
#   3. FCW 결과 CSV MD5 == GOLDEN_MD5
#   4. raw_frame_log.csv 존재
#   5. run_summary.json 존재
#
# quality 는 위와 별개로 측정 조건이 흔들렸는지를 봄
#   - 부하가 높은 상태에서 시작했는지
#   - 클럭 상태가 Run 도중 바뀌었는지
#   - tegrastats 가 있는데도 로그가 비었는지

run_one() {
    local run_index="$1"

    local run_id
    run_id="$(run_id_for "${run_index}")"

    local run_dir
    run_dir="$(printf '%s/run_%02d' "${SESSION_DIR}" "${run_index}")"
    mkdir -p "${run_dir}"

    local stdout_log="${run_dir}/stdout_stderr.log"
    local run_log_dir="${RUNS_ROOT_REL}/${run_id}"

    printf '  run id  : %s\n' "${run_id}"
    printf '  run dir : %s\n' "${run_dir}"

    # 이전 Run 이 남긴 결과 CSV 를 지움
    # 남겨 두면 ADAS 가 실패했을 때 앞 Run 의 파일을 검증해 잘못 PASS 가 됨
    rm -f "${FCW_CSV_REL}"

    # 배경 작업이 가라앉을 때까지 기다림 (측정 구간 밖)
    local idle_ok=1
    wait_for_idle || idle_ok=0

    # 시작 전 장비 상태 (측정 구간 밖에서 기록함)
    record_device_state "${run_dir}/device_before.txt" "before"
    snapshot_clocks "${run_dir}/clocks_before.txt"

    # 온도 기록 시작. 바이너리보다 먼저 띄워야 초반 구간이 안 빠짐
    start_tegrastats "${run_dir}/tegrastats.log"

    local started_at
    started_at="$(date '+%Y-%m-%d %H:%M:%S')"
    local start_seconds="${SECONDS}"

    # tegrastats 로그와 프레임 로그를 맞출 기준 시각
    date +%s.%N > "${run_dir}/tegrastats.log.anchor"

    printf '  started : %s\n' "${started_at}"

    # 인자는 배열로 만듦. 조건부 플래그를 문자열로 이어 붙이면 빈 값에서 깨짐
    local bin_args=(
        "${INPUT_REL}"
        --backend "${BACKEND}"
        --power-mode "${POWER_MODE}"
        --warmup-frames "${WARMUP_FRAMES}"
        --measured-frames "${MEASURED_FRAMES}"
        --run-id "${run_id}"
        --deadline-ms "${DEADLINE_MS}"
    )

    if [[ "${VIDEO_MODE}" == "off" ]]; then
        bin_args+=(--no-video)
    fi

    # set -e 가 켜져 있어도 || 왼쪽의 실패는 중단 사유가 아님
    # 실패해도 FAIL 로 기록하고 다음 Run 으로 가야 하므로 종료 코드만 받음
    local exit_code=0
    "./${BINARY_REL}" "${bin_args[@]}" > "${stdout_log}" 2>&1 || exit_code=$?

    local elapsed_s=$((SECONDS - start_seconds))

    # 온도 기록 종료 (측정 구간이 끝난 직후)
    stop_tegrastats

    # 종료 직후 장비 상태
    record_device_state "${run_dir}/device_after.txt" "after"
    snapshot_clocks "${run_dir}/clocks_after.txt"

    printf '  elapsed : %d s\n' "${elapsed_s}"
    printf '  exit    : %d\n' "${exit_code}"

    # 클럭 상태가 Run 도중 바뀌었는지 확인
    # 열이 올라 EDP 제한이 들어오면 --show 출력이 달라짐
    local clocks_changed=0
    if ! diff -q "${run_dir}/clocks_before.txt" "${run_dir}/clocks_after.txt" >/dev/null 2>&1; then
        clocks_changed=1
        printf '  [WARN] Run 도중 클럭 상태가 바뀜\n' >&2
    fi

    # tegrastats 로그 줄 수
    local tegrastats_lines=0
    if [[ -f "${run_dir}/tegrastats.log" ]]; then
        tegrastats_lines="$(wc -l < "${run_dir}/tegrastats.log" | tr -d ' ')"
    fi

    # FCW 결과 CSV 수집과 MD5 비교
    local fcw_csv_exists=0
    local fcw_csv_md5="none"
    local md5_match=0

    if [[ -f "${FCW_CSV_REL}" ]]; then
        fcw_csv_exists=1
        fcw_csv_md5="$(md5sum "${FCW_CSV_REL}" | awk '{print $1}')"
        cp "${FCW_CSV_REL}" "${run_dir}/fcw_frames.csv"

        if [[ "${fcw_csv_md5}" == "${GOLDEN_MD5}" ]]; then
            md5_match=1
        fi
    fi

    # RunLogger 산출물 수집
    local raw_frame_log_exists=0
    local run_summary_exists=0

    if [[ -f "${run_log_dir}/raw_frame_log.csv" ]]; then
        raw_frame_log_exists=1
        cp "${run_log_dir}/raw_frame_log.csv" "${run_dir}/raw_frame_log.csv"
    fi

    if [[ -f "${run_log_dir}/run_summary.json" ]]; then
        run_summary_exists=1
        cp "${run_log_dir}/run_summary.json" "${run_dir}/run_summary.json"
    fi

    # MD5 기록
    {
        printf 'run_id         : %s\n' "${run_id}"
        printf 'fcw_csv        : %s\n' "${FCW_CSV_REL}"
        printf 'golden_md5     : %s\n' "${GOLDEN_MD5}"
        printf 'fcw_csv_md5    : %s\n' "${fcw_csv_md5}"
        printf 'md5_match      : %s\n' "${md5_match}"
    } > "${run_dir}/md5.txt"

    # 판정 (정확성)
    local result="PASS"

    if [[ "${exit_code}" -ne 0 ]]; then
        result="FAIL"
        printf '  [FAIL] 종료 코드가 0 이 아님: %d\n' "${exit_code}" >&2
    fi

    if [[ "${fcw_csv_exists}" -ne 1 ]]; then
        result="FAIL"
        printf '  [FAIL] FCW 결과 CSV 없음: %s\n' "${FCW_CSV_REL}" >&2
    elif [[ "${md5_match}" -ne 1 ]]; then
        result="FAIL"
        printf '  [FAIL] FCW CSV MD5 불일치\n' >&2
        printf '         기대: %s\n' "${GOLDEN_MD5}" >&2
        printf '         실제: %s\n' "${fcw_csv_md5}" >&2
    fi

    if [[ "${raw_frame_log_exists}" -ne 1 ]]; then
        result="FAIL"
        printf '  [FAIL] raw_frame_log.csv 없음: %s\n' "${run_log_dir}" >&2
    fi

    if [[ "${run_summary_exists}" -ne 1 ]]; then
        result="FAIL"
        printf '  [FAIL] run_summary.json 없음: %s\n' "${run_log_dir}" >&2
    fi

    # 판정 (측정 조건)
    # 사유를 덮어쓰지 않고 모음. 하나만 남기면 뒤엣것이 앞엣것을 지워버림
    local quality_reasons=()

    if [[ "${idle_ok}" -ne 1 ]]; then
        quality_reasons+=("load_high")
    fi

    if [[ "${clocks_changed}" -eq 1 ]]; then
        quality_reasons+=("clocks_changed")
    fi

    # tegrastats 자체가 없는 x86 은 정상임. 있는데 로그가 빈 경우만 의심함
    if tegrastats_available && [[ "${tegrastats_lines}" -eq 0 ]]; then
        quality_reasons+=("no_tegrastats")
    fi

    local quality="OK"
    local quality_reason_text="none"

    if [[ "${#quality_reasons[@]}" -gt 0 ]]; then
        quality="SUSPECT"
        # CSV 안이라 쉼표를 못 씀. 파이프로 이음
        quality_reason_text="$(IFS='|'; echo "${quality_reasons[*]}")"
    fi

    # Run 단위 측정 환경을 결과 옆에 같이 남김
    # 이 파일만 보면 어떤 조건에서 나온 숫자인지 알 수 있음
    cat > "${run_dir}/measure_env.json" <<JSON
{
  "run_id": "${run_id}",
  "build_type": "${BUILD_TYPE}",
  "cxx_flags_release": "${CXX_FLAGS_RELEASE}",
  "backend": "${BACKEND}",
  "power_mode": "${POWER_MODE}",
  "power_mode_source": "${POWER_MODE_SOURCE}",
  "clocks_mode": "${CLOCKS_MODE}",
  "clocks_changed": ${clocks_changed},
  "video_mode": "${VIDEO_MODE}",
  "result_disk": "${RESULT_DISK}",
  "result_disk_kind": "${RESULT_DISK_KIND}",
  "warmup_frames": ${WARMUP_FRAMES},
  "measured_frames": ${MEASURED_FRAMES},
  "deadline_ms": ${DEADLINE_MS},
  "deadline_target": "${DEADLINE_TARGET}",
  "tegrastats_interval_ms": ${TEGRASTATS_INTERVAL_MS},
  "tegrastats_lines": ${tegrastats_lines},
  "idle_loadavg_max": ${IDLE_LOADAVG_MAX},
  "idle_gate_passed": ${idle_ok},
  "elapsed_s": ${elapsed_s},
  "exit_code": ${exit_code},
  "result": "${result}",
  "quality": "${quality}",
  "quality_reasons": "${quality_reason_text}"
}
JSON

    # tegrastats 로그 요약
    # 프레임 로그를 복사한 뒤에 돌려야 둘을 같은 시간축으로 묶을 수 있음
    if [[ "${tegrastats_lines}" -gt 0 ]] \
        && command -v python3 >/dev/null 2>&1 \
        && [[ -f "${TEGRASTATS_PARSER_REL}" ]]; then
        python3 "${TEGRASTATS_PARSER_REL}" "${run_dir}" || \
            printf '  [WARN] tegrastats 요약 실패 — 원본 로그는 남아 있음\n' >&2
    fi

    # 판정 근거를 컬럼으로 분리해 기록함
    # 자유 텍스트를 넣으면 쉼표 이스케이프 문제가 생기므로 쓰지 않음
    printf '%d,%s,%s,%d,%d,%d,%s,%d,%d,%d,%s,%s,%s,%d,%d,%d\n' \
        "${run_index}" \
        "${run_id}" \
        "${started_at}" \
        "${elapsed_s}" \
        "${exit_code}" \
        "${fcw_csv_exists}" \
        "${fcw_csv_md5}" \
        "${md5_match}" \
        "${raw_frame_log_exists}" \
        "${run_summary_exists}" \
        "${result}" \
        "${quality}" \
        "${quality_reason_text}" \
        "${tegrastats_lines}" \
        "${clocks_changed}" \
        "${idle_ok}" \
        >> "${SESSION_SUMMARY}"

    printf '  result  : %s\n' "${result}"
    printf '  quality : %s (%s)\n' "${quality}" "${quality_reason_text}"

    if [[ "${result}" == "PASS" ]]; then
        return 0
    fi
    return 1
}


# ------------------------------------------------------------
# Run 반복
# ------------------------------------------------------------

pass_count=0
fail_count=0
suspect_count=0

for ((run_index = 1; run_index <= RUN_COUNT; run_index++)); do
    printf '===== Run %d / %d =====\n' "${run_index}" "${RUN_COUNT}"

    # set -e 가 켜져 있어도 if 조건 안의 실패는 중단 사유가 아님
    if run_one "${run_index}"; then
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
    echo

    if (( run_index < RUN_COUNT )); then
        printf '[WAIT] %d 초 대기 (재개 예정 %s)\n' \
            "${WAIT_SECONDS}" \
            "$(date -d "+${WAIT_SECONDS} seconds" '+%H:%M:%S' 2>/dev/null || echo 'unknown')"
        sleep "${WAIT_SECONDS}"
        echo
    fi
done

# SUSPECT 개수는 요약 CSV 에서 셈
suspect_count="$(awk -F, 'NR > 1 && $12 == "SUSPECT"' "${SESSION_SUMMARY}" | wc -l | tr -d ' ')"


# ------------------------------------------------------------
# Session Manifest
# ------------------------------------------------------------
# 측정 조건을 결과물 옆에 같이 남김
# 압축을 풀었을 때 이 파일만 보면 어떤 조건이었는지 알 수 있음

readonly SESSION_MANIFEST="${SESSION_DIR}/session_manifest.txt"

model_md5="$(md5sum "${MODEL_REL}" | awk '{print $1}')"
input_md5="$(md5sum "${INPUT_REL}" | awk '{print $1}')"

{
    echo "===== Benchmark Session Manifest ====="
    printf 'session_id        : %s\n' "${SESSION_ID}"
    printf 'experiment        : %s\n' "${EXPERIMENT_NAME}"
    printf 'created_at        : %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')"
    printf 'host              : %s\n' "$(uname -srm)"
    echo

    echo "----- 측정 조건 -----"
    printf 'binary            : %s\n' "${BINARY_REL}"
    printf 'input             : %s\n' "${INPUT_REL}"
    printf 'model             : %s\n' "${MODEL_REL}"
    printf 'backend           : %s\n' "${BACKEND}"
    printf 'power_mode        : %s\n' "${POWER_MODE}"
    printf 'power_mode_source : %s\n' "${POWER_MODE_SOURCE}"
    printf 'warmup_frames     : %s\n' "${WARMUP_FRAMES}"
    printf 'measured_frames   : %s\n' "${MEASURED_FRAMES}"
    printf 'total_frames      : %s\n' "$((WARMUP_FRAMES + MEASURED_FRAMES))"
    printf 'run_count         : %s\n' "${RUN_COUNT}"
    printf 'wait_seconds      : %s\n' "${WAIT_SECONDS}"
    echo

    echo "----- 빌드 -----"
    printf 'build_type        : %s\n' "${BUILD_TYPE}"
    printf 'cxx_flags_release : %s\n' "${CXX_FLAGS_RELEASE}"
    echo

    echo "----- 측정 환경 -----"
    printf 'clocks_mode       : %s\n' "${CLOCKS_MODE}"
    printf 'video_mode        : %s\n' "${VIDEO_MODE}"
    printf 'result_disk       : %s\n' "${RESULT_DISK}"
    printf 'result_disk_kind  : %s\n' "${RESULT_DISK_KIND}"
    printf 'idle_loadavg_max  : %s\n' "${IDLE_LOADAVG_MAX}"
    printf 'idle_wait_max_s   : %s\n' "${IDLE_WAIT_MAX_S}"
    printf 'tegrastats        : %s\n' \
        "$(tegrastats_available && echo "on (${TEGRASTATS_INTERVAL_MS} ms)" || echo "unavailable")"
    echo

    echo "----- 평가 기준 (최적화 전 고정) -----"
    printf 'deadline_ms       : %s\n' "${DEADLINE_MS}"
    printf 'deadline_target   : %s\n' "${DEADLINE_TARGET}"
    echo

    echo "----- 입력 무결성 -----"
    printf 'golden_md5        : %s\n' "${GOLDEN_MD5}"
    printf 'model_md5         : %s\n' "${model_md5}"
    printf 'input_md5         : %s\n' "${input_md5}"
    echo

    echo "----- Git (하네스 실행 시점) -----"
    if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
        printf 'head              : %s\n' "$(git rev-parse HEAD)"

        if [[ -z "$(git status --porcelain --untracked-files=no)" ]]; then
            printf 'tracked_changes   : none\n'
        else
            printf 'tracked_changes   : present\n'
        fi
    else
        printf 'head              : unknown\n'
        printf 'tracked_changes   : unknown\n'
    fi
    echo

    echo "----- 판정 -----"
    printf 'pass              : %d / %d\n' "${pass_count}" "${RUN_COUNT}"
    printf 'fail              : %d / %d\n' "${fail_count}" "${RUN_COUNT}"
    printf 'suspect           : %s / %d (측정 조건이 흔들린 Run)\n' "${suspect_count}" "${RUN_COUNT}"
    echo

    echo "----- 파일 MD5 (세션 디렉터리 기준) -----"
    # 괄호 안은 별도 프로세스라 여기서 cd 해도 바깥 위치는 그대로임
    # manifest 자신은 아직 쓰는 중이므로 목록에서 제외함
    (
        cd "${SESSION_DIR}"
        find . -type f ! -name 'session_manifest.txt' | sort | while read -r file_path; do
            md5sum "${file_path}"
        done
    )
} > "${SESSION_MANIFEST}"


# ------------------------------------------------------------
# 압축
# ------------------------------------------------------------
# -C 로 세션 루트에 들어간 뒤 묶어서
# 압축 파일 안 경로가 <session_id>/... 로 시작하게 함

readonly ARCHIVE_PATH="${ARCHIVES_ROOT_REL}/${SESSION_ID}.tar.gz"

tar -czf "${ARCHIVE_PATH}" -C "${SESSIONS_ROOT_REL}" "${SESSION_ID}"

archive_md5="$(md5sum "${ARCHIVE_PATH}" | awk '{print $1}')"

# grep -c 는 0건일 때 실패로 끝나므로 || true 로 받음
archive_entries="$(tar -tzf "${ARCHIVE_PATH}" | grep -c -v '/$' || true)"


# ------------------------------------------------------------
# Session 결과
# ------------------------------------------------------------

echo "===== Session 결과 ====="
printf '  session id  : %s\n' "${SESSION_ID}"
printf '  session dir : %s\n' "${SESSION_DIR}"
printf '  summary     : %s\n' "${SESSION_SUMMARY}"
printf '  manifest    : %s\n' "${SESSION_MANIFEST}"
printf '  archive     : %s\n' "${ARCHIVE_PATH}"
printf '  archive md5 : %s\n' "${archive_md5}"
printf '  archive 파일: %s 개\n' "${archive_entries}"
printf '  PASS        : %d / %d\n' "${pass_count}" "${RUN_COUNT}"
printf '  FAIL        : %d / %d\n' "${fail_count}" "${RUN_COUNT}"
printf '  SUSPECT     : %s / %d\n' "${suspect_count}" "${RUN_COUNT}"
echo

if [[ "${suspect_count}" -ne 0 ]]; then
    echo "[NOTE] 측정 조건이 흔들린 Run 이 ${suspect_count} 개 있음 — session_summary.csv 의 quality 열 확인" >&2
fi

if [[ "${fail_count}" -eq 0 ]]; then
    echo "[PASS] 전체 ${RUN_COUNT} Run PASS"
    exit 0
fi

echo "[FAIL] ${fail_count} / ${RUN_COUNT} Run 실패" >&2
exit 1