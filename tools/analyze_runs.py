#!/usr/bin/env python3
"""
raw_frame_log.csv / run_summary.json 을 읽어 요약표를 출력한다.

사용법:
    python3 tools/analyze_runs.py results/runs/A results/runs/B ...
    python3 tools/analyze_runs.py results/runs            # 하위 run 디렉토리 전부

옵션:
    --warmup N          warmup 프레임 수를 run_summary.json 값 대신 N 으로
    --deadline-ms X     deadline 을 run_summary.json 값 대신 X 로
    --include-dirty     git_dirty run 도 포함 (기본 제외)
    --short-segment-s S LEAD 유지 구간을 "짧다"고 볼 기준 초 (기본 0.5)
    --csv PATH          run 별 요약표를 CSV 로도 저장

표준 라이브러리만 사용한다. 젯슨 로그도 같은 스키마(schema_version 1)면 그대로 읽는다.
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

SUPPORTED_SCHEMAS = {1, 2}


# ---------- 유틸 ----------

def percentile(values, p):
    """선형 보간 백분위 (numpy 기본과 같은 방식). 빈 목록이면 None."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def to_float(text):
    return float(text) if text not in ("", None) else None


def to_int(text):
    return int(text) if text not in ("", None) else None


def fmt(value, digits=1):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# ---------- run 읽기 ----------

def find_run_dirs(paths):
    runs = []
    for path in paths:
        if os.path.isfile(os.path.join(path, "run_summary.json")):
            runs.append(path)
            continue
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                candidate = os.path.join(path, name)
                if os.path.isfile(os.path.join(candidate, "run_summary.json")):
                    runs.append(candidate)
    return runs


def load_run(run_dir):
    with open(os.path.join(run_dir, "run_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    schema_version = summary.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMAS:
        supported = ", ".join(str(v) for v in sorted(SUPPORTED_SCHEMAS))
        raise ValueError(
            f"{run_dir}: schema_version {schema_version} 은 지원하지 않음 "
            f"(지원: {supported})"
        )
    with open(os.path.join(run_dir, "raw_frame_log.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return summary, rows


# ---------- 지표 ----------

def lead_segments(rows, fps):
    """연속으로 같은 lead_id(-1 아님)가 유지된 구간 목록. 장면 전환에서 끊긴 구간은 scene_cut=True."""
    segments = []
    current_id = None
    length = 0
    for row in rows:
        lead = to_int(row.get("lead_id"))
        scene_changed = to_int(row.get("scene_changed")) == 1
        if lead is None or lead == -1 or scene_changed:
            if current_id is not None:
                segments.append({"lead_id": current_id, "frames": length, "seconds": length / fps, "scene_cut": scene_changed})
            current_id = None
            length = 0
            if lead is None or lead == -1:
                continue
        if lead != current_id:
            if current_id is not None:
                segments.append({"lead_id": current_id, "frames": length, "seconds": length / fps, "scene_cut": False})
            current_id = lead
            length = 0
        length += 1
    if current_id is not None:
        segments.append({"lead_id": current_id, "frames": length, "seconds": length / fps, "scene_cut": False})
    return segments


def summarize_run(summary, rows, warmup, deadline_ms, short_segment_s):
    measured = rows[warmup:]
    fps_source = float(summary.get("source_fps") or 0.0) or 30.0
    result = {
        "run_id": summary.get("run_id"),
        "git": (summary.get("git_commit") or "")[:12] + ("*" if summary.get("git_dirty") else ""),
        "backend": summary.get("backend"),
        "precision": summary.get("precision"),
        "hardware": summary.get("hardware"),
        "frames": len(measured),
        "warmup": warmup,
        "deadline_ms": deadline_ms,
    }
    if not measured:
        return result

    def column(name):
        return [v for v in (to_float(r.get(name)) for r in measured) if v is not None]

    for name in ("total_processing_ms", "detect_ms", "inference_full_ms", "inference_crop_ms", "capture_ms", "output_ms"):
        values = column(name)
        result[f"{name}_p50"] = percentile(values, 50)
        result[f"{name}_p95"] = percentile(values, 95)
        result[f"{name}_max"] = max(values) if values else None

    # 합산 지표 (스키마의 원자값에서 파생)
    pre = [a + b for a, b in zip(column("preprocess_full_ms"), column("preprocess_crop_ms"))]
    post = [a + b + c for a, b, c in zip(column("postprocess_full_ms"), column("postprocess_crop_ms"), column("merge_ms"))]
    result["preprocess_ms_p50"] = percentile(pre, 50)
    result["postprocess_ms_p50"] = percentile(post, 50)

    # FPS: 측정 프레임의 dequeue 간격 기준 (처리량)
    dequeue = [to_int(r["dequeue_ts_ns"]) for r in measured]
    span_s = (dequeue[-1] - dequeue[0]) / 1e9 if len(dequeue) > 1 else 0.0
    result["fps"] = (len(dequeue) - 1) / span_s if span_s > 0 else None

    total = column("total_processing_ms")
    result["deadline_miss_rate"] = sum(1 for v in total if v > deadline_ms) / len(total) if total else None

    # Frame Age at Decision: capture 가 monotonic 축일 때만 정의
    ages = []
    for r in measured:
        if r.get("capture_ts_clock") == "monotonic":
            ages.append((to_int(r["decision_ts_ns"]) - to_int(r["capture_ts_ns"])) / 1e6)
    result["frame_age_ms_p50"] = percentile(ages, 50)
    result["frame_age_ms_p95"] = percentile(ages, 95)

    # Frame Drop: frame_seq 건너뜀
    seqs = [to_int(r["frame_seq"]) for r in measured]
    result["frame_drops"] = sum(max(0, b - a - 1) for a, b in zip(seqs, seqs[1:]))

    # FCW Task Metric (lead 컬럼이 있는 run 만)
    has_fcw = any(r.get("lead_id") not in ("", None) for r in measured)
    if has_fcw:
        segments = lead_segments(measured, fps_source)
        natural = [s for s in segments if not s["scene_cut"]]
        result["lead_segments"] = len(segments)
        result["lead_segment_median_s"] = statistics.median([s["seconds"] for s in segments]) if segments else None
        result["lead_short_segment_ratio"] = (sum(1 for s in natural if s["seconds"] < short_segment_s) / len(natural)) if natural else None
        lead_frames = [r for r in measured if to_int(r.get("lead_found")) == 1]
        with_ttc = [r for r in lead_frames if r.get("ttc_p") not in ("", None)]
        result["lead_frames"] = len(lead_frames)
        result["ttc_rate"] = len(with_ttc) / len(lead_frames) if lead_frames else None
        result["warning_frames"] = sum(1 for r in measured if r.get("warning_state") in ("CAUTION", "DANGER"))
    return result


# ---------- 출력 ----------

PER_RUN_COLUMNS = [
    ("run_id", "run", 0),
    ("git", "git", 0),
    ("frames", "frames", 0),
    ("fps", "FPS", 2),
    ("total_processing_ms_p50", "proc p50", 1),
    ("total_processing_ms_p95", "proc p95", 1),
    ("detect_ms_p50", "detect p50", 1),
    ("detect_ms_p95", "detect p95", 1),
    ("inference_full_ms_p50", "inf_full p50", 1),
    ("inference_crop_ms_p50", "inf_crop p50", 1),
    ("preprocess_ms_p50", "pre p50", 2),
    ("postprocess_ms_p50", "post p50", 2),
    ("deadline_miss_rate", "miss rate", 3),
    ("frame_age_ms_p50", "age p50", 1),
    ("frame_age_ms_p95", "age p95", 1),
    ("frame_drops", "drops", 0),
    ("lead_segments", "LEAD seg", 0),
    ("lead_segment_median_s", "seg med s", 2),
    ("lead_short_segment_ratio", "short ratio", 3),
    ("ttc_rate", "TTC rate", 3),
    ("warning_frames", "warn frames", 0),
]

CROSS_RUN_METRICS = [
    ("fps", "FPS", 2),
    ("total_processing_ms_p50", "proc p50 ms", 1),
    ("total_processing_ms_p95", "proc p95 ms", 1),
    ("detect_ms_p50", "detect p50 ms", 1),
    ("deadline_miss_rate", "miss rate", 3),
    ("lead_segment_median_s", "LEAD seg median s", 2),
    ("lead_short_segment_ratio", "short seg ratio", 3),
    ("ttc_rate", "TTC rate", 3),
]


def print_table(headers, rows):
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="run 디렉토리 또는 그 부모 디렉토리")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--deadline-ms", type=float, default=None)
    parser.add_argument("--include-dirty", action="store_true")
    parser.add_argument("--short-segment-s", type=float, default=0.5)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.paths)
    if not run_dirs:
        print("run 디렉토리를 찾지 못함 (run_summary.json 이 있는 디렉토리를 지정)", file=sys.stderr)
        return 1

    results = []
    skipped = []
    for run_dir in run_dirs:
        summary, rows = load_run(run_dir)
        if summary.get("git_dirty") and not args.include_dirty:
            skipped.append(summary.get("run_id"))
            continue
        warmup = args.warmup if args.warmup is not None else int(summary.get("warmup_frames") or 0)
        deadline = args.deadline_ms if args.deadline_ms is not None else float(summary.get("deadline_ms") or 0.0)
        results.append(summarize_run(summary, rows, warmup, deadline, args.short_segment_s))

    if skipped:
        print(f"[제외] git dirty run {len(skipped)}개: {', '.join(skipped)}  (--include-dirty 로 포함)\n")
    if not results:
        print("분석할 run 이 없음", file=sys.stderr)
        return 1

    print("== run 별 요약 (warmup 제외) ==")
    headers = [label for _, label, _ in PER_RUN_COLUMNS]
    table = [[fmt(r.get(key), digits) for key, _, digits in PER_RUN_COLUMNS] for r in results]
    print_table(headers, table)
    print()
    print("deadline_ms: " + ", ".join(f"{r['run_id']}={fmt(r['deadline_ms'], 1)}" for r in results))
    print("hardware: " + "; ".join(sorted({r["hardware"] or "unknown" for r in results})))
    print("frame age: capture_ts_clock == monotonic 인 run 에서만 계산 (영상 파일은 n/a)")

    if len(results) > 1:
        print("\n== run 간 변동 (min / max / (max-min)/median) ==")
        table = []
        for key, label, digits in CROSS_RUN_METRICS:
            values = [r.get(key) for r in results if r.get(key) is not None]
            if not values:
                continue
            low, high, med = min(values), max(values), statistics.median(values)
            spread = (high - low) / med if med else None
            table.append([label, fmt(low, digits), fmt(high, digits), fmt(spread * 100 if spread is not None else None, 1) + ("%" if spread is not None else "")])
        print_table(["metric", "min", "max", "spread"], table)

    if args.csv:
        keys = [key for key, _, _ in PER_RUN_COLUMNS]
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for r in results:
                writer.writerow(["" if r.get(k) is None else r.get(k) for k in keys])
        print(f"\n요약 CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
