#!/usr/bin/env bash
# 평가셋 실행 루프
# eval/videos.csv 의 각 줄을 영상별 lane ROI 로 돌리고
# _frames / _banner / run_summary 를 results/eval/<backend>/ 에 모음
# 사용: bash scripts/run_eval_set.sh [목록csv] [backend]
set -euo pipefail

LIST="${1:-eval/videos.csv}"
BACKEND="${2:-tensorrt_fp16}"
VIDEO_DIR="${VIDEO_DIR:-videos/eval}"
POWER_MODE="${POWER_MODE:-MAXN_SUPER}"
OUT_DIR="results/eval/${BACKEND}"
mkdir -p "${OUT_DIR}"

# 헤더 건너뜀, 빈 줄·# 줄 무시, Windows 줄바꿈 제거
tail -n +2 "${LIST}" | tr -d '\r' | grep -v '^[[:space:]]*$' | grep -v '^#' \
| while IFS=, read -r id type group should_warn collision_frame cf_status x1 y1 x2 y2 x3 y3 x4 y4; do
    video="${VIDEO_DIR}/${id}.mp4"
    if [[ ! -f "${video}" ]]; then
        echo "[SKIP] 영상 없음: ${video}" >&2
        continue
    fi
    if [[ -z "${x1}" || -z "${y4}" ]]; then
        echo "[SKIP] ROI 없음: ${id}" >&2
        continue
    fi
    roi="${x1},${y1},${x2},${y2},${x3},${y3},${x4},${y4}"
    run_id="eval_${id}_${BACKEND}"
    echo "[RUN] ${id} backend=${BACKEND} roi=${roi}"

    if ! ./build/apps/adas "${video}" --backend "${BACKEND}" --power-mode "${POWER_MODE}" \
            --deadline-ms 66.7 --run-id "${run_id}" --no-video --lane-roi "${roi}" \
            > "${OUT_DIR}/${id}.log" 2>&1; then
        echo "[FAIL] ${id} (로그: ${OUT_DIR}/${id}.log)" >&2
        continue
    fi

    cp "results/${id}_frames.csv"                  "${OUT_DIR}/${id}_frames.csv"
    cp "results/${id}_banner.csv"                  "${OUT_DIR}/${id}_banner.csv"
    cp "results/runs/${run_id}/run_summary.json"   "${OUT_DIR}/${id}_run_summary.json"
done

echo "[DONE] ${OUT_DIR}"