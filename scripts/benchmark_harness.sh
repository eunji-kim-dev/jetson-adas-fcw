#!/usr/bin/env bash

# 고정된 측정 조건으로 ADAS 를 반복 실행하고 결과를 검증하는 벤치마크 하네스
#
# 사용법:
#   bash scripts/benchmark_harness.sh <실험이름>
#   예: bash scripts/benchmark_harness.sh baseline_x86
#
# [1단계 골격] Preflight 검사와 Session / Run ID 생성까지만 수행함
#              ADAS 실행과 판정은 다음 단계에서 추가함

set -euo pipefail


# ------------------------------------------------------------
# 고정 측정 조건
# ------------------------------------------------------------

readonly RUN_COUNT=5
readonly WAIT_SECONDS=900
readonly WARMUP_FRAMES=15
readonly MEASURED_FRAMES=1237

readonly BACKEND="opencv_dnn"
readonly POWER_MODE="ac_balanced"

readonly BINARY_REL="build/apps/adas"
readonly INPUT_REL="videos/input.mp4"
readonly MODEL_REL="models/yolov8n.onnx"

readonly GOLDEN_REL="results/golden_baseline.csv"
readonly GOLDEN_MD5="a0006c4a16dbe3f69c178fbc5c1b6b8e"

readonly SESSIONS_ROOT_REL="results/benchmark_sessions"
readonly ARCHIVES_ROOT_REL="results/benchmark_archives"
readonly RUNS_ROOT_REL="results/runs"


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

if [[ $# -ne 1 ]]; then
    echo "사용법: bash scripts/benchmark_harness.sh <실험이름>" >&2
    echo "  예:   bash scripts/benchmark_harness.sh baseline_x86" >&2
    exit 2
fi

readonly EXPERIMENT_NAME="$1"

# 실험 이름은 그대로 디렉터리 이름이 되므로 공백과 경로 문자를 막음
if [[ ! "${EXPERIMENT_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[ERROR] 실험 이름에는 영문/숫자/밑줄/하이픈만 사용: ${EXPERIMENT_NAME}" >&2
    exit 2
fi


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
# Preflight
# ------------------------------------------------------------

echo "===== Preflight ====="
echo "  project root: ${PROJECT_ROOT}"
echo

echo "필요한 명령"
require_command md5sum
require_command tar
require_command date
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
printf '  power mode      : %s\n' "${POWER_MODE}"
printf '  warmup frames   : %s\n' "${WARMUP_FRAMES}"
printf '  measured frames : %s\n' "${MEASURED_FRAMES}"
printf '  total frames    : %s\n' "$((WARMUP_FRAMES + MEASURED_FRAMES))"
printf '  run count       : %s\n' "${RUN_COUNT}"
printf '  wait between    : %s s\n' "${WAIT_SECONDS}"
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

readonly SESSION_SUMMARY="${SESSION_DIR}/session_summary.csv"

printf 'run_index,run_id,started_at,elapsed_s,exit_code,fcw_csv_exists,fcw_csv_md5,md5_match,raw_frame_log_exists,run_summary_exists,result\n' \
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

    local started_at
    started_at="$(date '+%Y-%m-%d %H:%M:%S')"
    local start_seconds="${SECONDS}"

    printf '  started : %s\n' "${started_at}"

    # set -e 가 켜져 있어도 || 왼쪽의 실패는 중단 사유가 아님
    # 실패해도 FAIL 로 기록하고 다음 Run 으로 가야 하므로 종료 코드만 받음
    local exit_code=0
    "./${BINARY_REL}" \
        "${INPUT_REL}" \
        --backend "${BACKEND}" \
        --power-mode "${POWER_MODE}" \
        --warmup-frames "${WARMUP_FRAMES}" \
        --measured-frames "${MEASURED_FRAMES}" \
        --run-id "${run_id}" \
        > "${stdout_log}" 2>&1 || exit_code=$?

    local elapsed_s=$((SECONDS - start_seconds))

    printf '  elapsed : %d s\n' "${elapsed_s}"
    printf '  exit    : %d\n' "${exit_code}"

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

    # 판정
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

    # 판정 근거를 컬럼으로 분리해 기록함
    # 자유 텍스트를 넣으면 쉼표 이스케이프 문제가 생기므로 쓰지 않음
    printf '%d,%s,%s,%d,%d,%d,%s,%d,%d,%d,%s\n' \
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
        >> "${SESSION_SUMMARY}"

    printf '  result  : %s\n' "${result}"

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
    printf 'warmup_frames     : %s\n' "${WARMUP_FRAMES}"
    printf 'measured_frames   : %s\n' "${MEASURED_FRAMES}"
    printf 'total_frames      : %s\n' "$((WARMUP_FRAMES + MEASURED_FRAMES))"
    printf 'run_count         : %s\n' "${RUN_COUNT}"
    printf 'wait_seconds      : %s\n' "${WAIT_SECONDS}"
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
echo

if [[ "${fail_count}" -eq 0 ]]; then
    echo "[PASS] 전체 ${RUN_COUNT} Run PASS"
    exit 0
fi

echo "[FAIL] ${fail_count} / ${RUN_COUNT} Run 실패" >&2
exit 1