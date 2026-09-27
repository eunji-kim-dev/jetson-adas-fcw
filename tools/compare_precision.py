#!/usr/bin/env python3
"""
Q5 정밀도 비교 — 평가셋 결과 폴더 두 개(기준 vs 비교)를 박스·추적 단위로 비교함

사용법:
    python3 compare_precision.py results/eval/tensorrt_fp32_crop288x640 results/eval/tensorrt_fp16_crop288x640 \
        --videos eval/videos.csv --label-a fp32 --label-b fp16 --out results/eval/q5_fp32_vs_fp16

입력: 각 폴더의 <id>_frames.csv (frame,sceneChanged,numDetections,detections,numTracks,tracks,activeLeadId,riskLevel,ttc)
      detections = cls:x:y:w:h:conf;...   tracks = id:cls:x:y:w:h;...

지표 (9/24 640 vs 288 비교 방식 그대로):
    1. 짝 맞은 박스(IoU>=0.5)의 높이 차 분포 — 0px / ±1px / ±2px 비율
    2. 한쪽에만 있는 박스 — 개수, 높이 중앙값, 20px 이하 비율, 신뢰도 중앙값
    3. LEAD 같은 차량 비율 — 두 쪽 다 LEAD 가 있는 프레임에서 LEAD 박스 IoU>=0.5
    4. LEAD ID 바뀐 횟수 — 충돌 전 구간(collision_frame 까지, 없으면 전체)
    5. 박스 아래 변(y+h) 차 분포 — 호모그래피용 (10/8 부터 봄)

출력:
    <out>_summary.csv   영상별 한 줄 + 전체 한 줄
    <out>_lead.csv      프레임별 LEAD 박스 높이·아래 변 (05 처럼 시계열을 볼 때)
    <out>_only.csv      한쪽에만 있는 박스 목록 (19·21 처럼 특정 물체를 찾을 때)
표준 라이브러리만 씀
"""

import argparse
import csv
import os
import statistics
import sys


def parse_boxes(text, with_conf):
    boxes = []
    if not text:
        return boxes
    for token in text.split(";"):
        parts = token.split(":")
        if with_conf:
            cls, x, y, w, h, conf = parts
            boxes.append({"cls": int(cls), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "conf": float(conf)})
        else:
            tid, cls, x, y, w, h = parts
            boxes.append({"id": int(tid), "cls": int(cls), "x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    return boxes


def iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def match(boxes_a, boxes_b, threshold=0.5):
    """IoU 큰 순으로 1:1 짝 맞춤. (짝 목록, A 만, B 만)"""
    pairs = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            v = iou(a, b)
            if v >= threshold:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_a, used_b, matched = set(), set(), []
    for v, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((boxes_a[i], boxes_b[j], v))
    only_a = [a for i, a in enumerate(boxes_a) if i not in used_a]
    only_b = [b for j, b in enumerate(boxes_b) if j not in used_b]
    return matched, only_a, only_b


def load_frames(folder, video_id):
    path = os.path.join(folder, f"{video_id}_frames.csv")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    frames = {}
    for r in rows:
        frames[int(r["frame"])] = {
            "det": parse_boxes(r["detections"], True),
            "trk": parse_boxes(r["tracks"], False),
            "lead": int(r["activeLeadId"]),
        }
    return frames


def lead_box(frame):
    if frame["lead"] < 0:
        return None
    for t in frame["trk"]:
        if t["id"] == frame["lead"]:
            return t
    return None


def median(values):
    return statistics.median(values) if values else None


def fmt(v, d=1):
    if v is None:
        return ""
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder_a", help="기준 (FP32)")
    p.add_argument("folder_b", help="비교 (FP16 또는 INT8)")
    p.add_argument("--videos", default="eval/videos.csv")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out", required=True, help="출력 파일 접두어")
    p.add_argument("--small-px", type=int, default=20, help="'작은 박스' 기준 높이 (기본 20px)")
    p.add_argument("--min-conf", type=float, default=0.0, help="이 신뢰도 미만 검출 박스는 양쪽 다 빼고 비교 (기본 0 = 전부)")
    args = p.parse_args()

    with open(args.videos, encoding="utf-8", newline="") as f:
        videos = list(csv.DictReader(f))

    la, lb = args.label_a, args.label_b
    summary_rows = []
    lead_rows = []
    only_rows = []
    total = {"matched": 0, "h0": 0, "h1": 0, "h2": 0, "b1": 0, "b2": 0, "only_a": [], "only_b": [], "conf_diff": [],
             "lead_both": 0, "lead_same": 0, "lead_only_a": 0, "lead_only_b": 0, "lead_change_a": 0, "lead_change_b": 0,
             "frames": 0, "det_diff": 0}

    for v in videos:
        vid = v["id"]
        try:
            fa = load_frames(args.folder_a, vid)
            fb = load_frames(args.folder_b, vid)
        except FileNotFoundError as e:
            print(f"[SKIP] {vid}: {e}", file=sys.stderr)
            continue
        cf = int(v["collision_frame"]) if v.get("collision_frame") else None
        limit = cf if cf else max(fa)

        st = {"matched": 0, "h0": 0, "h1": 0, "h2": 0, "b1": 0, "b2": 0, "only_a": [], "only_b": [], "conf_diff": [],
              "lead_both": 0, "lead_same": 0, "lead_only_a": 0, "lead_only_b": 0, "frames": 0, "det_diff": 0}
        prev_lead_a = prev_lead_b = None
        change_a = change_b = 0

        for fr in sorted(fa):
            if fr not in fb:
                continue
            A, B = fa[fr], fb[fr]
            st["frames"] += 1
            # --min-conf: 약한 박스(문턱 근처)를 빼고 보면 양자화 잡음과 진짜 누락이 갈림
            det_a = [d for d in A["det"] if d["conf"] >= args.min_conf]
            det_b = [d for d in B["det"] if d["conf"] >= args.min_conf]
            if len(det_a) != len(det_b):
                st["det_diff"] += 1

            matched, only_a, only_b = match(det_a, det_b)
            st["matched"] += len(matched)
            for a, b, _ in matched:
                dh = abs(a["h"] - b["h"])
                db = abs((a["y"] + a["h"]) - (b["y"] + b["h"]))
                st["h0"] += dh == 0
                st["h1"] += dh <= 1
                st["h2"] += dh <= 2
                st["b1"] += db <= 1
                st["b2"] += db <= 2
                st["conf_diff"].append(b["conf"] - a["conf"])   # 같은 차를 잡았을 때 B 신뢰도 - A 신뢰도
            st["only_a"] += only_a
            st["only_b"] += only_b
            for side, boxes in ((la, only_a), (lb, only_b)):
                for bx in boxes:
                    only_rows.append([vid, fr, side, bx["cls"], bx["x"], bx["y"], bx["w"], bx["h"], f"{bx['conf']:.3f}"])

            # LEAD
            lead_a, lead_b = lead_box(A), lead_box(B)
            if lead_a and lead_b:
                st["lead_both"] += 1
                if iou(lead_a, lead_b) >= 0.5:
                    st["lead_same"] += 1
            elif lead_a:
                st["lead_only_a"] += 1
            elif lead_b:
                st["lead_only_b"] += 1
            lead_rows.append([vid, fr,
                              A["lead"], lead_a["h"] if lead_a else "", (lead_a["y"] + lead_a["h"]) if lead_a else "",
                              B["lead"], lead_b["h"] if lead_b else "", (lead_b["y"] + lead_b["h"]) if lead_b else ""])

            # 충돌 전 구간 LEAD ID 변경 (LEAD 없음(-1) 은 세지 않고, 다른 ID 로 바뀔 때만)
            if fr <= limit:
                if A["lead"] >= 0 and prev_lead_a is not None and A["lead"] != prev_lead_a:
                    change_a += 1
                if B["lead"] >= 0 and prev_lead_b is not None and B["lead"] != prev_lead_b:
                    change_b += 1
                if A["lead"] >= 0:
                    prev_lead_a = A["lead"]
                if B["lead"] >= 0:
                    prev_lead_b = B["lead"]

        def row(name, s, ca, cb):
            m = s["matched"] or 1
            oa, ob = s["only_a"], s["only_b"]
            lb_ = s["lead_both"] or 1
            return [name, s["frames"], s["det_diff"], s["matched"],
                    fmt(100 * s["h0"] / m), fmt(100 * s["h1"] / m), fmt(100 * s["h2"] / m),
                    fmt(100 * s["b1"] / m), fmt(100 * s["b2"] / m),
                    len(oa), fmt(median([b["h"] for b in oa])), fmt(100 * sum(b["h"] <= args.small_px for b in oa) / len(oa)) if oa else "", fmt(median([b["conf"] for b in oa]), 2),
                    len(ob), fmt(median([b["h"] for b in ob])), fmt(100 * sum(b["h"] <= args.small_px for b in ob) / len(ob)) if ob else "", fmt(median([b["conf"] for b in ob]), 2),
                    s["lead_both"], fmt(100 * s["lead_same"] / lb_), s["lead_only_a"], s["lead_only_b"], ca, cb,
                    fmt(median(s["conf_diff"]), 3), fmt(100 * sum(d <= -0.1 for d in s["conf_diff"]) / m)]

        summary_rows.append(row(vid, st, change_a, change_b))
        for k in ("matched", "h0", "h1", "h2", "b1", "b2", "lead_both", "lead_same", "lead_only_a", "lead_only_b", "frames", "det_diff"):
            total[k] += st[k]
        total["only_a"] += st["only_a"]
        total["only_b"] += st["only_b"]
        total["conf_diff"] += st["conf_diff"]
        total["lead_change_a"] += change_a
        total["lead_change_b"] += change_b

    summary_rows.append(row("전체", total, total["lead_change_a"], total["lead_change_b"]))

    header = ["video", "frames", "det_count_diff_frames", "matched",
              "h_diff_0px_%", "h_diff_le1px_%", "h_diff_le2px_%", "bottom_diff_le1px_%", "bottom_diff_le2px_%",
              f"only_{la}", f"only_{la}_h_median", f"only_{la}_small_%", f"only_{la}_conf_median",
              f"only_{lb}", f"only_{lb}_h_median", f"only_{lb}_small_%", f"only_{lb}_conf_median",
              "lead_both_frames", "lead_same_vehicle_%", f"lead_only_{la}", f"lead_only_{lb}",
              f"lead_id_changes_precollision_{la}", f"lead_id_changes_precollision_{lb}",
              f"conf_{lb}_minus_{la}_median", f"conf_drop_ge0.1_%"]

    with open(args.out + "_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)
    with open(args.out + "_lead.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", f"lead_id_{la}", f"lead_h_{la}", f"lead_bottom_{la}", f"lead_id_{lb}", f"lead_h_{lb}", f"lead_bottom_{lb}"])
        w.writerows(lead_rows)
    with open(args.out + "_only.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame", "only_in", "cls", "x", "y", "w", "h", "conf"])
        w.writerows(only_rows)

    # 화면 출력: 핵심 열만
    print(f"== {la} vs {lb} (기준 {la}, 신뢰도 >= {args.min_conf}) ==")
    cols = [0, 1, 2, 3, 5, 6, 8, 9, 10, 13, 14, 17, 18, 21, 22, 23, 24]
    short = ["video", "frames", "detΔ", "matched", "h≤1px%", "h≤2px%", "bot≤2px%",
             f"only_{la}", "h_med", f"only_{lb}", "h_med", "lead_both", "same%", f"idchg_{la}", f"idchg_{lb}", "confΔmed", "conf↓0.1%"]
    rows = [[str(r[c]) for c in cols] for r in summary_rows]
    widths = [max(len(short[i]), *(len(r[i]) for r in rows)) for i in range(len(cols))]
    print("  ".join(h.ljust(w) for h, w in zip(short, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(f"\n저장: {args.out}_summary.csv, {args.out}_lead.csv, {args.out}_only.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())