#!/usr/bin/env python3
"""raw_frame_log.csv 를 읽어서 deadline 기준 지표를 냄.

표준 라이브러리만 씀.

사용법:
    python3 tools/deadline_report.py results/runs/run01 results/runs/run02 ...
    python3 tools/deadline_report.py results/benchmark_sessions/<세션>/

deadline 값과 대상 컬럼은 아래 상수로 고정함.
최적화 실험 뒤에 이 값을 고치면 비교가 깨지므로 고칠 때는 작업일지에 남겨야 함.
"""

import csv
import glob
import json
import os
import sys

# ------------------------------------------------------------
# 사전 고정 상수
# ------------------------------------------------------------

DEADLINE_MS = 66.7          # 15 FPS 기준 1프레임 예산
DEADLINE_TARGET = "processing_ms"   # 카메라 붙이면 frame_age_ms 로 바꿀 자리
TRAILING_RATIO = 0.20       # 안정 구간으로 보는 뒤쪽 비율
STREAK_ALERT = 3            # 연속 초과가 이 개수 이상이면 따로 셈


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def pick_column(fieldnames):
    lowered = {f.lower(): f for f in fieldnames}
    if DEADLINE_TARGET in lowered:
        return lowered[DEADLINE_TARGET]
    for c in ("processing_ms", "total_ms", "pipeline_ms", "frame_ms", "latency_ms"):
        if c in lowered:
            return lowered[c]
    return None


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []
    col = pick_column(fields)
    if col is None:
        raise SystemExit("처리시간 컬럼을 못 찾음. 실제 컬럼=%s" % fields)
    vals = []
    for r in rows:
        try:
            vals.append(float(r[col]))
        except (TypeError, ValueError):
            continue
    return vals, col


def streaks(values, budget):
    """예산을 연속으로 넘긴 구간들의 길이 목록."""
    out = []
    cur = 0
    for v in values:
        if v > budget:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def analyze(values, label):
    n = len(values)
    if n == 0:
        return None
    misses = [v for v in values if v > DEADLINE_MS]
    st = streaks(values, DEADLINE_MS)
    p50 = pct(values, 50)
    p95 = pct(values, 95)
    p99 = pct(values, 99)
    return {
        "window": label,
        "frames": n,
        "miss_count": len(misses),
        "miss_rate_pct": round(100.0 * len(misses) / n, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max(values), 2),
        "p50_over_budget_x": round(p50 / DEADLINE_MS, 2),
        "p95_over_budget_x": round(p95 / DEADLINE_MS, 2),
        "max_overrun_ms": round(max(values) - DEADLINE_MS, 2),
        "max_streak": max(st) if st else 0,
        "streaks_over_%d" % STREAK_ALERT: sum(1 for s in st if s >= STREAK_ALERT),
        "effective_fps": round(1000.0 / p50, 2),
        "headroom_needed_x": round(p95 / DEADLINE_MS, 2),
    }


def run_dirs(args):
    out = []
    for a in args:
        if os.path.isfile(os.path.join(a, "raw_frame_log.csv")):
            out.append(a)
        else:
            out.extend(sorted(
                os.path.dirname(p)
                for p in glob.glob(os.path.join(a, "**", "raw_frame_log.csv"),
                                   recursive=True)))
    return out


def main():
    if len(sys.argv) < 2:
        print("사용법: python3 tools/deadline_report.py <run 디렉터리...>", file=sys.stderr)
        return 2

    dirs = run_dirs(sys.argv[1:])
    if not dirs:
        print("raw_frame_log.csv 를 가진 디렉터리가 없음", file=sys.stderr)
        return 1

    report = {
        "deadline_ms": DEADLINE_MS,
        "deadline_target": DEADLINE_TARGET,
        "trailing_ratio": TRAILING_RATIO,
        "runs": {},
    }

    print("deadline %.1f ms (15 FPS 기준 1프레임 예산), 대상 = %s"
          % (DEADLINE_MS, DEADLINE_TARGET))
    print("")
    header = "%-20s %-12s %9s %9s %8s %9s %7s" % (
        "run", "구간", "p50ms", "예산대비", "miss%", "최대연속", "fps")
    print(header)
    print("-" * len(header))

    for d in dirs:
        vals, col = load(os.path.join(d, "raw_frame_log.csv"))
        cut = int(len(vals) * (1 - TRAILING_RATIO))
        entries = [
            analyze(vals, "full"),
            analyze(vals[cut:], "trailing%d" % int(TRAILING_RATIO * 100)),
        ]
        entries = [e for e in entries if e]
        report["runs"][os.path.basename(d.rstrip("/"))] = {
            "column_used": col,
            "windows": entries,
        }
        for e in entries:
            print("%-20s %-12s %9.1f %8.1fx %8.1f %9d %7.2f" % (
                os.path.basename(d.rstrip("/")) if e["window"] == "full" else "",
                e["window"], e["p50_ms"], e["p50_over_budget_x"],
                e["miss_rate_pct"], e["max_streak"], e["effective_fps"]))

    # 회차 간 편차 — 안정 구간 p50 기준
    trailing = []
    for v in report["runs"].values():
        for e in v["windows"]:
            if e["window"].startswith("trailing"):
                trailing.append(e["p50_ms"])
    if len(trailing) > 1:
        spread = (max(trailing) - min(trailing)) / min(trailing) * 100.0
        report["trailing_p50_spread_pct"] = round(spread, 2)
        print("")
        print("안정구간 p50 %.1f~%.1f ms, 회차 간 변동폭 %.1f%%"
              % (min(trailing), max(trailing), spread))

    out_path = "deadline_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("→ %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
