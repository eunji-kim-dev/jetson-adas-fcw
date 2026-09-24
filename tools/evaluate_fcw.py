#!/usr/bin/env python3
"""
평가셋 배너 결과(<id>_banner.csv)와 정답표(eval/videos.csv)로
Warning Miss / False Alarm / 경고 선행시간을 계산한다.

사용법:
    python3 tools/evaluate_fcw.py results/eval/tensorrt_fp16
    python3 tools/evaluate_fcw.py results/eval/tensorrt_fp16 results/eval/tensorrt_fp16_crop288x640   # 두 구조 나란히

옵션:
    --videos PATH       정답표 (기본 eval/videos.csv)
    --warn-level LEVEL  경고로 볼 배너 단계 (기본 DANGER, CAUTION 을 주면 CAUTION 이상)
    --fps N             선행시간 계산용 fps. run_summary 에 source_fps 가 있으면 그 값을 우선 씀 (기본 15)
    --csv PATH          영상별 표를 CSV 로도 저장

판정 규칙:
    - 경고 = 배너가 --warn-level 이상으로 뜬 것 (기본 DANGER)
    - Warning Miss  = should_warn=1 인데 collision_frame 전까지 경고가 없음. 충돌 이후에 뜬 경고는 늦은 경고라 Miss 로 셈
    - False Alarm   = should_warn=0 인데 경고가 한 번이라도 뜸
    - 선행시간(s)   = (collision_frame - 첫 경고 프레임) / fps
    - collision_frame 이 없는 영상(cf_status=none)은 경고 여부만 보고 선행시간은 비움
    - 요약은 선행시간을 "confirmed 만" 과 "전체(estimated 포함)" 두 줄로 냄

표준 라이브러리만 사용한다.
"""

import argparse
import csv
import json
import os
import statistics
import sys

LEVEL_ORDER = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}


# ---------- 입력 읽기 ----------

def read_videos(path):
    """정답표를 읽어 id 순서 목록으로 돌려줌. Windows 줄바꿈과 빈 줄은 무시함"""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            if not row.get("id"):
                continue
            rows.append({
                "id": row["id"],
                "type": row.get("type", ""),
                "group": row.get("group", ""),
                "should_warn": int(row["should_warn"]) if row.get("should_warn") else None,
                "collision_frame": int(row["collision_frame"]) if row.get("collision_frame") else None,
                "cf_status": row.get("cf_status", "") or ("none" if not row.get("collision_frame") else "confirmed"),
            })
    return rows


def read_banner(path):
    """<id>_banner.csv → [(frame, level, lead_id, ttc_text), ...]"""
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append((int(row["frame"]), row["bannerLevel"].strip(), row.get("activeLeadId", ""), row.get("ttc", "")))
    return out


def read_source_fps(result_dir, video_id, default_fps):
    """<id>_run_summary.json 의 source_fps. 없으면 기본값"""
    path = os.path.join(result_dir, f"{video_id}_run_summary.json")
    if not os.path.exists(path):
        return default_fps
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f).get("source_fps")
        return float(value) if value else default_fps
    except (OSError, ValueError, json.JSONDecodeError):
        return default_fps


# ---------- 판정 ----------

def evaluate_video(video, banner, fps, warn_threshold):
    """영상 하나의 판정 결과 dict"""
    warn_frames = [fr for fr, lvl, _, _ in banner if LEVEL_ORDER.get(lvl, 0) >= warn_threshold]
    caution_frames = [fr for fr, lvl, _, _ in banner if LEVEL_ORDER.get(lvl, 0) >= LEVEL_ORDER["CAUTION"]]
    first_warn = warn_frames[0] if warn_frames else None
    first_caution = caution_frames[0] if caution_frames else None
    cf = video["collision_frame"]

    result = {
        "id": video["id"],
        "group": video["group"],
        "should_warn": video["should_warn"],
        "collision_frame": cf,
        "cf_status": video["cf_status"],
        "frames": len(banner),
        "first_caution": first_caution,
        "first_warn": first_warn,
        "warn_frames": len(warn_frames),
        "miss": None,
        "false_alarm": None,
        "late": False,
        "lead_time_s": None,
    }

    if video["should_warn"] == 1:
        if first_warn is None:
            result["miss"] = True
        elif cf is None:
            # 충돌 프레임이 없으면 경고 여부만 판단함
            result["miss"] = False
        elif first_warn >= cf:
            # 충돌 프레임 이후의 경고는 늦은 경고 → Miss
            result["miss"] = True
            result["late"] = True
        else:
            result["miss"] = False
            result["lead_time_s"] = (cf - first_warn) / fps
    elif video["should_warn"] == 0:
        result["false_alarm"] = first_warn is not None

    return result


def summarize(results):
    """그룹별·전체 요약. 선행시간은 confirmed 만 / 전체 두 가지"""
    def block(rows):
        targets = [r for r in rows if r["should_warn"] == 1]
        non_targets = [r for r in rows if r["should_warn"] == 0]
        lead_all = [r["lead_time_s"] for r in targets if r["lead_time_s"] is not None]
        lead_conf = [r["lead_time_s"] for r in targets if r["lead_time_s"] is not None and r["cf_status"] == "confirmed"]
        return {
            "n": len(rows),
            "targets": len(targets),
            "miss": sum(1 for r in targets if r["miss"]),
            "late": sum(1 for r in targets if r["late"]),
            "non_targets": len(non_targets),
            "false_alarm": sum(1 for r in non_targets if r["false_alarm"]),
            "lead_conf_n": len(lead_conf),
            "lead_conf_mean": statistics.mean(lead_conf) if lead_conf else None,
            "lead_conf_median": statistics.median(lead_conf) if lead_conf else None,
            "lead_all_n": len(lead_all),
            "lead_all_mean": statistics.mean(lead_all) if lead_all else None,
            "lead_all_median": statistics.median(lead_all) if lead_all else None,
        }

    groups = []
    for g in sorted({r["group"] for r in results}):
        groups.append((g, block([r for r in results if r["group"] == g])))
    groups.append(("전체", block(results)))
    return groups


# ---------- 출력 ----------

def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "O" if value else "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def verdict(r):
    """영상 한 줄의 판정 글자"""
    if r["should_warn"] == 1:
        if r["late"]:
            return "MISS(늦음)"
        return "MISS" if r["miss"] else "OK"
    if r["should_warn"] == 0:
        return "FA" if r["false_alarm"] else "OK"
    return "?"


def print_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def print_per_video(labels, per_dir):
    """영상별 표. 결과 폴더가 여러 개면 나란히 놓고, 판정이 다르면 *, 첫 경고 프레임만 다르면 ~ 표시"""
    headers = ["id", "group", "warn", "cf", "cf_status"]
    for label in labels:
        headers += [f"{label} 판정", f"{label} 첫경고", f"{label} 선행s"]
    if len(labels) > 1:
        headers.append("차이")

    rows = []
    ids = [r["id"] for r in per_dir[labels[0]]]
    by_id = {label: {r["id"]: r for r in per_dir[label]} for label in labels}
    for vid in ids:
        base = by_id[labels[0]][vid]
        row = [vid, base["group"], fmt(base["should_warn"]), fmt(base["collision_frame"]), base["cf_status"]]
        verdicts = []
        for label in labels:
            r = by_id[label].get(vid)
            if r is None:
                row += ["(없음)", "-", "-"]
                verdicts.append(None)
                continue
            row += [verdict(r), fmt(r["first_warn"]), fmt(r["lead_time_s"])]
            verdicts.append((verdict(r), r["first_warn"]))
        if len(labels) > 1:
            # * = 판정(OK/MISS/FA)이 다름, ~ = 판정은 같고 첫 경고 프레임만 다름
            kinds = {v[0] if v else None for v in verdicts}
            frames = {v[1] if v else None for v in verdicts}
            row.append("*" if len(kinds) > 1 else ("~" if len(frames) > 1 else ""))
        rows.append(row)
    print_table(headers, rows)


def print_summary(labels, per_dir):
    for label in labels:
        print(f"\n[{label}]")
        headers = ["그룹", "영상", "대상", "Miss", "(늦음)", "비대상", "FA",
                   "선행 conf n", "conf 평균", "conf 중앙", "선행 전체 n", "전체 평균", "전체 중앙"]
        rows = []
        for name, b in summarize(per_dir[label]):
            rows.append([name, b["n"], b["targets"], b["miss"], b["late"], b["non_targets"], b["false_alarm"],
                         b["lead_conf_n"], fmt(b["lead_conf_mean"]), fmt(b["lead_conf_median"]),
                         b["lead_all_n"], fmt(b["lead_all_mean"]), fmt(b["lead_all_median"])])
        print_table(headers, rows)


def write_csv(path, labels, per_dir):
    fields = ["result", "id", "group", "should_warn", "collision_frame", "cf_status", "frames",
              "first_caution", "first_warn", "warn_frames", "verdict", "miss", "late", "false_alarm", "lead_time_s"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for label in labels:
            for r in per_dir[label]:
                writer.writerow([label, r["id"], r["group"], r["should_warn"], r["collision_frame"], r["cf_status"],
                                 r["frames"], r["first_caution"], r["first_warn"], r["warn_frames"], verdict(r),
                                 r["miss"], r["late"], r["false_alarm"],
                                 "" if r["lead_time_s"] is None else f"{r['lead_time_s']:.3f}"])


# ---------- 메인 ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dirs", nargs="+", help="results/eval/<backend> 폴더 (여러 개면 나란히 비교)")
    parser.add_argument("--videos", default="eval/videos.csv")
    parser.add_argument("--warn-level", default="DANGER", choices=["CAUTION", "DANGER"])
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    videos = read_videos(args.videos)
    warn_threshold = LEVEL_ORDER[args.warn_level]

    labels = []
    per_dir = {}
    for result_dir in args.result_dirs:
        label = os.path.basename(os.path.normpath(result_dir))
        labels.append(label)
        results = []
        for video in videos:
            banner_path = os.path.join(result_dir, f"{video['id']}_banner.csv")
            if not os.path.exists(banner_path):
                print(f"[WARN] 배너 파일 없음: {banner_path}", file=sys.stderr)
                continue
            fps = read_source_fps(result_dir, video["id"], args.fps)
            results.append(evaluate_video(video, read_banner(banner_path), fps, warn_threshold))
        per_dir[label] = results

    print(f"정답표 {args.videos} ({len(videos)}개), 경고 기준 = 배너 {args.warn_level} 이상\n")
    print_per_video(labels, per_dir)
    print_summary(labels, per_dir)

    if args.csv:
        write_csv(args.csv, labels, per_dir)
        print(f"\nCSV 저장: {args.csv}")


if __name__ == "__main__":
    main()