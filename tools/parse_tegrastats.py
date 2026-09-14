#!/usr/bin/env python3
"""tegrastats 로그를 읽어서 요약 JSON 과 타임라인 CSV 로 바꿈.

표준 라이브러리만 씀. Jetson 에 pandas 안 깔려 있어도 돌아감.

사용법:
    python3 tools/parse_tegrastats.py results/runs/<run_id>

run 디렉터리 안에 있어야 하는 파일:
    tegrastats.log          tegrastats --logfile 출력
    tegrastats.log.anchor   하네스가 남긴 기준 시각 (binary_start epoch)
    raw_frame_log.csv       (선택) 있으면 시간축을 맞춰서 같이 묶음

만드는 파일:
    tegrastats_summary.json
    tegrastats_timeline.csv
    tegrastats_frame_joined.csv   (raw_frame_log.csv 가 있을 때만)
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime

# ------------------------------------------------------------
# 판정 기준 — 측정 전에 고정하는 값들
# ------------------------------------------------------------

# Orin Nano 는 tj 가 이 근처부터 소프트웨어 스로틀이 들어감
TJ_WARN_C = 80.0
TJ_THROTTLE_C = 95.0

# 클럭이 최고치 대비 이만큼 떨어지면 스로틀로 봄
CPU_FREQ_DROP_RATIO = 0.90

# 실행 직후 몇 초는 아직 클럭이 올라오는 중이라 하락 판정에서 뺌
FREQ_SETTLE_SKIP_S = 10


# ------------------------------------------------------------
# 한 줄 파싱
# ------------------------------------------------------------

RE_TS = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU_BLOCK = re.compile(r"CPU \[([^\]]+)\]")
RE_CPU_CORE = re.compile(r"(\d+)%@(\d+)")
RE_EMC = re.compile(r"EMC_FREQ (\d+)%")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%")
RE_GR3D_MHZ = re.compile(r"GR3D_FREQ \d+%@\[?(\d+)")
RE_TEMP = re.compile(r"\b([a-zA-Z0-9_]+)@(-?[\d.]+)C")
RE_RAIL = re.compile(r"\b(VDD[_A-Z0-9]*|VIN[_A-Z0-9]*) (\d+)mW/(\d+)mW")


def parse_line(line):
    """tegrastats 한 줄을 dict 로 바꿈. 못 읽으면 None 을 냄."""
    m = RE_TS.match(line)
    if not m:
        return None

    # tegrastats 는 로컬 시간으로 찍음
    dt = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
    row = {"epoch": time.mktime(dt.timetuple()), "wallclock": m.group(1)}

    m = RE_RAM.search(line)
    if m:
        row["ram_used_mb"] = int(m.group(1))
        row["ram_total_mb"] = int(m.group(2))

    m = RE_SWAP.search(line)
    if m:
        row["swap_used_mb"] = int(m.group(1))

    m = RE_CPU_BLOCK.search(line)
    if m:
        cores = RE_CPU_CORE.findall(m.group(1))
        if cores:
            utils = [int(u) for u, _ in cores]
            freqs = [int(f) for _, f in cores]
            row["cpu_util_mean_pct"] = sum(utils) / len(utils)
            row["cpu_util_max_pct"] = max(utils)
            row["cpu_freq_max_mhz"] = max(freqs)
            row["cpu_core_count"] = len(cores)

    m = RE_EMC.search(line)
    if m:
        row["emc_pct"] = int(m.group(1))

    m = RE_GR3D.search(line)
    if m:
        row["gr3d_pct"] = int(m.group(1))

    m = RE_GR3D_MHZ.search(line)
    if m:
        row["gr3d_freq_mhz"] = int(m.group(1))

    # 온도 존은 보드마다 이름이 달라서 전부 긁어옴
    for zone, val in RE_TEMP.findall(line):
        row["temp_" + zone.lower() + "_c"] = float(val)

    # 전력 레일도 마찬가지로 전부 긁어옴. 앞의 값이 순간값, 뒤가 평균값
    for rail, inst, avg in RE_RAIL.findall(line):
        key = rail.lower()
        row["pwr_" + key + "_mw"] = int(inst)
        row["pwr_" + key + "_avg_mw"] = int(avg)

    return row


# ------------------------------------------------------------
# 통계 도우미
# ------------------------------------------------------------

def col(rows, key):
    return [r[key] for r in rows if key in r and r[key] is not None]


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


def summarize(rows):
    out = {"sample_count": len(rows)}
    if not rows:
        return out

    out["duration_s"] = rows[-1]["epoch"] - rows[0]["epoch"]

    # 온도 존별 최대/평균
    temp_keys = sorted({k for r in rows for k in r if k.startswith("temp_")})
    out["temps"] = {}
    for k in temp_keys:
        v = col(rows, k)
        if v:
            out["temps"][k] = {
                "max_c": round(max(v), 2),
                "mean_c": round(sum(v) / len(v), 2),
                "start_c": round(v[0], 2),
                "end_c": round(v[-1], 2),
            }

    # 전력
    rail_keys = sorted({k for r in rows for k in r
                        if k.startswith("pwr_") and not k.endswith("_avg_mw")})
    out["power"] = {}
    for k in rail_keys:
        v = col(rows, k)
        if v:
            mean_mw = sum(v) / len(v)
            out["power"][k] = {
                "max_mw": max(v),
                "mean_mw": round(mean_mw, 1),
            }
    # VDD_IN 이 있으면 대략적인 에너지도 같이 냄
    vin = col(rows, "pwr_vdd_in_mw")
    if vin and out.get("duration_s"):
        out["power"]["energy_j_estimate"] = round(
            sum(vin) / len(vin) / 1000.0 * out["duration_s"], 1)

    # GPU / CPU
    for key, label in (("gr3d_pct", "gr3d"), ("cpu_util_mean_pct", "cpu_util"),
                       ("cpu_freq_max_mhz", "cpu_freq_mhz"),
                       ("ram_used_mb", "ram_used_mb")):
        v = col(rows, key)
        if v:
            out[label] = {
                "min": round(min(v), 1),
                "max": round(max(v), 1),
                "mean": round(sum(v) / len(v), 1),
                "p95": round(pct(v, 95), 1),
            }

    # 스로틀 신호
    flags = {}
    tj = col(rows, "temp_tj_c") or col(rows, "temp_cpu_c")
    if tj:
        flags["tj_max_c"] = round(max(tj), 2)
        flags["samples_over_warn"] = sum(1 for t in tj if t >= TJ_WARN_C)
        flags["samples_over_throttle"] = sum(1 for t in tj if t >= TJ_THROTTLE_C)
        flags["thermal_warn"] = flags["samples_over_warn"] > 0
        flags["thermal_throttle"] = flags["samples_over_throttle"] > 0

    t0 = rows[0]["epoch"]
    settled = [r for r in rows if r["epoch"] - t0 >= FREQ_SETTLE_SKIP_S]
    f = col(settled or rows, "cpu_freq_max_mhz")
    if f:
        peak = max(f)
        low = min(f)
        flags["cpu_freq_peak_mhz"] = peak
        flags["cpu_freq_min_mhz"] = low
        flags["cpu_freq_window"] = "실행 %d초 이후" % FREQ_SETTLE_SKIP_S
        flags["cpu_freq_dropped"] = bool(peak and low < peak * CPU_FREQ_DROP_RATIO)
    out["throttle_flags"] = flags
    return out


# ------------------------------------------------------------
# 프레임 로그와 시간축 맞추기
# ------------------------------------------------------------

FRAME_TIME_CANDIDATES = [
    "total_processing_ms", "processing_ms", "total_ms", "detect_ms",
    "latency_ms", "elapsed_ms", "proc_ms",
]
FRAME_SEQ_CANDIDATES = ["frame_seq", "frame_id", "frame_index", "seq"]
FRAME_TS_CANDIDATES = ["capture_ts", "capture_ts_ms", "ts_ms", "wall_ts"]


def pick(fieldnames, candidates):
    lowered = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def load_frames(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []
    return rows, fields


def join_frames(rows, frame_path, anchor_epoch, out_path):
    """1초 단위로 프레임 처리시간과 온도를 같은 표에 올림."""
    frames, fields = load_frames(frame_path)
    if not frames:
        return None

    tcol = pick(fields, FRAME_TIME_CANDIDATES)
    scol = pick(fields, FRAME_SEQ_CANDIDATES)
    if tcol is None:
        return {"joined": False, "reason":
                "처리시간 컬럼을 못 찾음. 후보=%s, 실제=%s" % (FRAME_TIME_CANDIDATES, fields)}

    # 프레임별 경과시간은 처리시간 누적으로 근사함
    # (capture_ts 가 실제 벽시계면 그걸 쓰는 게 정확하지만 소스에 따라 가짜 값이라 누적을 씀)
    elapsed = 0.0
    buckets = {}
    for fr in frames:
        try:
            ms = float(fr[tcol])
        except (TypeError, ValueError):
            continue
        elapsed += ms / 1000.0
        b = int(elapsed)
        buckets.setdefault(b, []).append(ms)

    # tegrastats 쪽도 같은 초 단위로
    tegra_by_sec = {}
    for r in rows:
        b = int(r["epoch"] - anchor_epoch)
        if b >= 0:
            tegra_by_sec[b] = r

    tj_key = "temp_tj_c" if any("temp_tj_c" in r for r in rows) else "temp_cpu_c"

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["elapsed_s", "frame_count", "frame_ms_p50",
                    "tj_c", "gpu_c", "gr3d_pct", "cpu_freq_mhz",
                    "cpu_util_mean_pct", "vdd_in_mw"])
        for b in sorted(buckets):
            t = tegra_by_sec.get(b, {})
            w.writerow([
                b,
                len(buckets[b]),
                round(pct(buckets[b], 50), 2),
                t.get(tj_key, ""),
                t.get("temp_gpu_c", ""),
                t.get("gr3d_pct", ""),
                t.get("cpu_freq_max_mhz", ""),
                round(t["cpu_util_mean_pct"], 1) if "cpu_util_mean_pct" in t else "",
                t.get("pwr_vdd_in_mw", ""),
            ])

    matched = sum(1 for b in buckets if b in tegra_by_sec)
    return {
        "joined": True,
        "frame_time_column": tcol,
        "frame_seq_column": scol,
        "bucket_count": len(buckets),
        "bucket_with_tegrastats": matched,
        "coverage_pct": round(100.0 * matched / len(buckets), 1) if buckets else 0.0,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("사용법: python3 tools/parse_tegrastats.py <run 디렉터리>", file=sys.stderr)
        return 2

    run_dir = sys.argv[1]
    log_path = os.path.join(run_dir, "tegrastats.log")
    anchor_path = os.path.join(run_dir, "tegrastats.log.anchor")
    frame_path = os.path.join(run_dir, "raw_frame_log.csv")

    if not os.path.isfile(log_path):
        print("tegrastats.log 없음: %s" % log_path, file=sys.stderr)
        return 1

    rows = []
    bad = 0
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = parse_line(line)
            if r is None:
                bad += 1
            else:
                rows.append(r)

    if not rows:
        print("파싱된 줄이 0개임. 로그 형식을 확인해야 함", file=sys.stderr)
        return 1

    summary = summarize(rows)
    summary["unparsed_lines"] = bad
    summary["source"] = os.path.abspath(log_path)

    # 타임라인 CSV
    anchor_epoch = rows[0]["epoch"]
    if os.path.isfile(anchor_path):
        with open(anchor_path, encoding="utf-8") as fh:
            try:
                anchor_epoch = float(fh.read().strip())
                summary["anchor"] = "binary_start"
            except ValueError:
                summary["anchor"] = "first_sample (anchor 파일 읽기 실패)"
    else:
        summary["anchor"] = "first_sample (anchor 파일 없음)"

    all_keys = sorted({k for r in rows for k in r})
    timeline_path = os.path.join(run_dir, "tegrastats_timeline.csv")
    with open(timeline_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["elapsed_s"] + all_keys)
        for r in rows:
            w.writerow([round(r["epoch"] - anchor_epoch, 1)] +
                       [r.get(k, "") for k in all_keys])

    # 프레임 로그와 묶기
    if os.path.isfile(frame_path):
        joined_path = os.path.join(run_dir, "tegrastats_frame_joined.csv")
        summary["frame_join"] = join_frames(rows, frame_path, anchor_epoch, joined_path)
    else:
        summary["frame_join"] = {"joined": False, "reason": "raw_frame_log.csv 없음"}

    summary_path = os.path.join(run_dir, "tegrastats_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # 사람이 보는 요약
    fl = summary.get("throttle_flags", {})
    print("샘플 %d개, %.0f초" % (summary["sample_count"], summary.get("duration_s", 0)))
    if "tj_max_c" in fl:
        print("tj 최고 %.1fC  (경고선 %.0fC, 스로틀선 %.0fC)"
              % (fl["tj_max_c"], TJ_WARN_C, TJ_THROTTLE_C))
    if "cpu_freq_peak_mhz" in fl:
        print("CPU 클럭 %d~%d MHz  하락판정=%s"
              % (fl["cpu_freq_min_mhz"], fl["cpu_freq_peak_mhz"],
                 "예" if fl.get("cpu_freq_dropped") else "아니오"))
    if "gr3d" in summary:
        print("GR3D %.0f~%.0f%% (평균 %.0f%%)"
              % (summary["gr3d"]["min"], summary["gr3d"]["max"], summary["gr3d"]["mean"]))
    pw = summary.get("power", {}).get("pwr_vdd_in_mw")
    if pw:
        print("VDD_IN 평균 %.0f mW, 최대 %d mW" % (pw["mean_mw"], pw["max_mw"]))
    print("→ %s" % summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
