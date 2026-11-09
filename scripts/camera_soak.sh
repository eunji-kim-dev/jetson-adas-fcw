#!/usr/bin/env bash
# 카메라 자원·장시간 측정 루프
# 카메라 입력(30fps 고정)으로 돌리면서 전력·온도·프로세스 메모리를 같이 적음
#   대기(프로그램 없음) → 네 경로 5분씩(CPU 전처리 / GPU v1 / GPU v2 / 최종) → 최종 설정 1시간 → 대기
# 사용: bash scripts/camera_soak.sh check|smoke|run [이름]
#   check  전력 모드·카메라 간격·골든 확인 (약 2분)
#   smoke  구간마다 10초씩만 돌려 흐름 확인 (약 3분)
#   run    실제 측정 (약 1시간 30분)
#   이름을 안 주면 soak<월일> 을 씀. check·smoke 는 뒤에 _check·_smoke 가 붙어 run 결과와 안 섞임
# 결과
#   results/<이름>/<구간>/        tegrastats.log, proc_mem.csv, run.log, tegrastats_summary.json
#   results/runs/<이름>_<구간>/   raw_frame_log.csv, run_summary.json
#   results/<이름>/session.log    화면에 찍힌 내용 전부
# 카메라는 측정 내내 같은 장면을 보게 고정함. 다른 실행과 같이 돌리지 않음

set -u

MODE="${1:-}"
NAME="${2:-soak$(date +%m%d)}"   # 결과 폴더 이름

BIN="./build/apps/adas"
PROBE="./build/apps/camera_probe"
CAM="/dev/video0"                # 카메라 장치
INPUT="videos/input.mp4"         # 골든 확인용 영상
RULES="overlap,history,gap,passby,gate,bonnet,hold,rank,bottom,edge,cropedge,rawconfirm"   # 최종 12규칙
MODEL_OPTS="--backend tensorrt_fp16 --crop-model models/yolov8n_288x640.onnx --crop-input 288x640 --lead-rule ${RULES}"
GOLDEN="results/golden_baseline_aarch64_tensorrt_fp16_crop288x640_lead12"   # 뒤에 .csv / _banner.csv 가 붙음

WARMUP=40                        # 앞 40프레임은 요약에서 빠짐 (카메라 시작 직후 멈춤 구간)
GAP_S=20                         # 구간 사이 쉬는 시간(초). 카메라가 닫히길 기다림

case "${MODE}" in
    check) TAG="${NAME}_check" ;;
    smoke) TAG="${NAME}_smoke"; PATH_FRAMES=300;  SOAK_FRAMES=300;    IDLE_S=10  ;;
    run)   TAG="${NAME}";       PATH_FRAMES=9000; SOAK_FRAMES=108000; IDLE_S=180 ;;   # 5분 / 1시간 / 3분
    *) echo "사용법: bash scripts/camera_soak.sh check|smoke|run [이름]"; exit 1 ;;
esac

OUT="results/${TAG}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/session.log") 2>&1

# ------------------------------------------------------------
# 준비
# ------------------------------------------------------------

# sudo 를 한 번 인증하고 1분마다 갱신함. 기본 15분이면 만료돼서 중간에 tegrastats 가 안 뜸
sudo -v || { echo "[중단] sudo 인증 실패"; exit 1; }
( while true; do sudo -n true 2>/dev/null || exit 0; sleep 60; kill -0 "$$" 2>/dev/null || exit 0; done ) >/dev/null 2>&1 &
KEEP_PID=$!

cleanup() {
    sudo -n tegrastats --stop >/dev/null 2>&1
    pkill -P "${KEEP_PID}" 2>/dev/null     # 갱신 루프가 물고 있는 sleep 까지 같이 끔
    kill "${KEEP_PID}" 2>/dev/null
}
trap cleanup EXIT

for f in "${BIN}" "${PROBE}" tools/parse_tegrastats.py tools/analyze_runs.py; do
    [ -e "${f}" ] || { echo "[중단] 없음: ${f} — 저장소 폴더에서 실행했는지 확인"; exit 1; }
done
[ -e "${CAM}" ] || { echo "[중단] 카메라 없음: ${CAM}"; exit 1; }

# 팬 세기 파일(0~255). 보드마다 경로가 달라서 못 찾으면 빈칸으로 둠
FAN_FILE="$(ls /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1 2>/dev/null | head -n 1)"

sudo -n jetson_clocks && echo "jetson_clocks 적용함" || echo "[확인] jetson_clocks 적용 실패 — 클럭이 고정 안 됐을 수 있음"
echo "전력 모드: $(sudo -n nvpmodel -q 2>/dev/null | head -n 1)"
echo "git: $(git rev-parse --short HEAD 2>/dev/null) / 수정된 파일 $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l)개"

# ------------------------------------------------------------
# 1초마다 메모리를 한 줄씩 적음
#   pid 가 "-" 면 보드 값만 적음 (대기 구간용)
#   limit_s 를 넘기면 멈춘 걸로 보고 프로그램을 끔
# ------------------------------------------------------------
sample_mem() {
    local pid="$1" file="$2" limit_s="${3:-0}"
    local now fan started=${SECONDS}
    echo "epoch,vm_rss_kb,vm_hwm_kb,vm_size_kb,threads,mem_available_kb,nvmap_used_kb,fan_pwm" > "${file}"
    while true; do
        if [ "${pid}" != "-" ] && ! kill -0 "${pid}" 2>/dev/null; then break; fi
        printf -v now '%(%s)T' -1
        fan=""
        if [ -n "${FAN_FILE}" ]; then read -r fan < "${FAN_FILE}" 2>/dev/null || fan=""; fi
        if [ "${pid}" = "-" ]; then
            awk -v t="${now}" -v fan="${fan}" '
                /^MemAvailable:/ {avail=$2} /^NvMapMemUsed:/ {nv=$2}
                END {printf "%s,,,,,%s,%s,%s\n", t, avail, nv, fan}' /proc/meminfo >> "${file}" 2>/dev/null
        else
            # VmRSS = 지금 쓰는 양, VmHWM = 지금까지 최고치
            awk -v t="${now}" -v fan="${fan}" '
                /^VmRSS:/ {rss=$2} /^VmHWM:/ {hwm=$2} /^VmSize:/ {size=$2} /^Threads:/ {thr=$2}
                /^MemAvailable:/ {avail=$2} /^NvMapMemUsed:/ {nv=$2}
                END {if (rss != "") printf "%s,%s,%s,%s,%s,%s,%s,%s\n", t, rss, hwm, size, thr, avail, nv, fan}' \
                "/proc/${pid}/status" /proc/meminfo >> "${file}" 2>/dev/null
            if [ "${limit_s}" -gt 0 ] && [ $(( SECONDS - started )) -gt "${limit_s}" ]; then
                echo "    [시간 초과] ${limit_s}초를 넘겨서 프로그램을 끔"
                kill "${pid}" 2>/dev/null
                sleep 3
                kill -9 "${pid}" 2>/dev/null
                break
            fi
        fi
        sleep 1
    done
}

start_tegrastats() {   # 기록 폴더
    sudo -n tegrastats --stop >/dev/null 2>&1
    rm -f "$1/tegrastats.log"
    sudo -n tegrastats --interval 1000 --logfile "$1/tegrastats.log" &
    sleep 2
}

stop_tegrastats() {    # 기록 폴더
    sudo -n tegrastats --stop
    sleep 1
    python3 tools/parse_tegrastats.py "$1" > "$1/parse.log" 2>&1 || echo "    [확인] parse_tegrastats 실패 — $1/parse.log"
}

# ------------------------------------------------------------
# 대기 구간: 프로그램 없이 보드만 켜 둔 상태
# ------------------------------------------------------------
run_idle() {   # 구간 이름, 초
    local name="$1" secs="$2"
    local dir="${OUT}/${name}"
    mkdir -p "${dir}"
    echo "[$(date +%H:%M:%S)] ${name} 시작 — ${secs}초"
    start_tegrastats "${dir}"
    sample_mem - "${dir}/proc_mem.csv" &
    local sampler=$!
    sleep "${secs}"
    kill "${sampler}" 2>/dev/null
    wait "${sampler}" 2>/dev/null
    stop_tegrastats "${dir}"
}

# ------------------------------------------------------------
# 카메라 구간: 프로그램을 돌리면서 tegrastats 와 메모리를 같이 적음
# ------------------------------------------------------------
run_seg() {   # 구간 이름, 측정 프레임 수, 경로별 옵션...
    local name="$1" frames="$2"; shift 2
    local dir="${OUT}/${name}"
    local run_id="${TAG}_${name}"
    local limit_s=$(( frames / 20 + 180 ))      # 20fps 로 떨어져도 끝날 시간 + 시작 여유 3분
    mkdir -p "${dir}"
    echo "[$(date +%H:%M:%S)] ${name} 시작 — ${frames}프레임 (옵션: ${*:-없음})"

    start_tegrastats "${dir}"
    # shellcheck disable=SC2086
    "${BIN}" --camera "${CAM}" ${MODEL_OPTS} "$@" \
        --warmup-frames "${WARMUP}" --measured-frames "${frames}" --no-video \
        --run-id "${run_id}" > "${dir}/run.log" 2>&1 &
    local pid=$!
    sample_mem "${pid}" "${dir}/proc_mem.csv" "${limit_s}"
    wait "${pid}"
    local code=$?
    stop_tegrastats "${dir}"

    # 카메라 결과 CSV 는 다음 구간이 덮어쓰므로 구간 폴더에 따로 둠
    cp results/video0_frames.csv results/video0_banner.csv "${dir}/" 2>/dev/null

    echo "    종료 코드 ${code} / ERROR $(grep -c 'ERROR' "${dir}/run.log") / $(grep '처리 프레임 수' "${dir}/run.log" | tail -n 1)"
    awk -F, 'NR > 1 && $2 != "" {last = $2; if ($3 > hwm) hwm = $3; n++}
             END {printf "    메모리(RSS) 끝 %.0f MB / 최고 %.0f MB / 기록 %d줄\n", last / 1024, hwm / 1024, n}' "${dir}/proc_mem.csv"

    if [ "${code}" -ne 0 ]; then
        echo "[중단] ${name} 이 실패함 — ${dir}/run.log 끝부분:"
        tail -n 15 "${dir}/run.log"
        exit 1
    fi
    sleep "${GAP_S}"
}

same_md5() {   # 파일 둘이 같은지
    if [ -e "$1" ] && [ -e "$2" ] && [ "$(md5sum < "$1")" = "$(md5sum < "$2")" ]; then echo "같음"; else echo "다름"; fi
}

# ------------------------------------------------------------
# check: 측정 전에 눈으로 볼 것들
# ------------------------------------------------------------
if [ "${MODE}" = "check" ]; then
    echo
    echo "== 다른 실행이 없어야 함 (아래 줄과 loadavg 사이가 비어 있으면 됨) =="
    pgrep -a -f "apps/adas|apps/camera_probe|benchmark_harness|run_eval_set"
    echo "loadavg: $(cat /proc/loadavg)"

    echo
    echo "== 카메라 간격 (촬영 간격 p50 이 32~33 ms 면 됨) =="
    "${PROBE}" "${CAM}" --frames 300

    echo
    echo "== 골든 확인 (영상 파일, 최종 설정, 12규칙) =="
    # shellcheck disable=SC2086
    "${BIN}" "${INPUT}" ${MODEL_OPTS} --gpu-preprocess --gpu-postprocess --no-video \
        --run-id "${TAG}_golden" > "${OUT}/golden.log" 2>&1
    echo "종료 코드 $? / ERROR $(grep -c 'ERROR' "${OUT}/golden.log")"
    echo "_frames : $(same_md5 results/input_frames.csv "${GOLDEN}.csv")   $(md5sum < results/input_frames.csv | cut -c1-8)"
    echo "_banner : $(same_md5 results/input_banner.csv "${GOLDEN}_banner.csv")   $(md5sum < results/input_banner.csv | cut -c1-8)"
    echo
    echo "전력 모드가 MAXN_SUPER, 카메라 간격 32~33 ms, 골든 둘 다 '같음' 이면 smoke 로 넘어감"
    exit 0
fi

# ------------------------------------------------------------
# smoke / run: 대기 → 네 경로 → 장시간 → 대기
# ------------------------------------------------------------

# 같은 보드에서 다른 실행과 겹치면 숫자가 흔들리므로 시작하지 않음
if pgrep -f "apps/adas|benchmark_harness|run_eval_set" > /dev/null; then
    echo "[중단] 다른 실행이 돌고 있음:"; pgrep -a -f "apps/adas|benchmark_harness|run_eval_set"; exit 1
fi

echo
echo "== ${MODE} 시작 $(date '+%Y-%m-%d %H:%M:%S') — 경로당 ${PATH_FRAMES}프레임, 장시간 ${SOAK_FRAMES}프레임 =="

run_idle idle_before "${IDLE_S}"

run_seg cpu    "${PATH_FRAMES}"                                                       # CPU 전처리
run_seg gpu_v1 "${PATH_FRAMES}" --gpu-preprocess                                      # GPU 전처리 v1
run_seg gpu_v2 "${PATH_FRAMES}" --gpu-preprocess --camera-zero-copy                   # GPU 전처리 v2
run_seg final  "${PATH_FRAMES}" --gpu-preprocess --camera-zero-copy --gpu-postprocess # 최종

run_seg soak   "${SOAK_FRAMES}" --gpu-preprocess --camera-zero-copy --gpu-postprocess # 최종 설정 장시간

run_idle idle_after "${IDLE_S}"     # 달궈진 상태의 대기 전력. 팬·열 몫을 가르는 데 씀

echo
echo "== 속도 요약 =="
python3 tools/analyze_runs.py "results/runs/${TAG}_cpu" "results/runs/${TAG}_gpu_v1" "results/runs/${TAG}_gpu_v2" \
    "results/runs/${TAG}_final" "results/runs/${TAG}_soak" || echo "[확인] analyze_runs 실패"

echo
echo "== 끝 $(date '+%Y-%m-%d %H:%M:%S') — 결과: ${OUT}/ , results/runs/${TAG}_* =="