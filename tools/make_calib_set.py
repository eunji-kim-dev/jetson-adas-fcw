#!/usr/bin/env python3
"""
INT8 calibration 이미지 세트를 만든다.

영상에서 일정 간격으로 프레임을 뽑아
  <out>/full/<영상stem>_<프레임>.jpg   전체 프레임      (전체 프레임 모델 640x640 용)
  <out>/crop/<영상stem>_<프레임>.jpg   원거리 crop 영역 (crop 모델 288x640 용)
으로 저장하고, 경로 목록 파일 두 개를 쓴다.
  <out>/full_list.txt
  <out>/crop_list.txt

사용법:
    python3 tools/make_calib_set.py --out calib videos/calib/*.mp4
    python3 tools/make_calib_set.py --out calib --per-video 20 videos/calib/*.mp4

옵션:
    --out DIR           출력 폴더 (기본 calib)
    --per-video N       영상 하나에서 뽑을 프레임 수 (기본 15). 영상 길이에 맞춰 균등 간격으로 뽑음
    --skip-head N       영상 앞쪽에서 건너뛸 프레임 수 (기본 5, 시작 직후 어두운 프레임 제외용)
    --quality Q         JPEG 품질 (기본 95)

crop 영역은 YoloDetector::detectFarRoadObjects 와 같음 (x 25%, y 38%, w 50%, h 36%).
평가셋 21개 영상은 여기에 넣지 말 것 (채점지로 calibration 하는 셈이 됨).
표준 라이브러리 + OpenCV(cv2)만 사용한다.
"""

import argparse
import os
import sys

try:
    import cv2
except ImportError:
    print("cv2(OpenCV python)가 필요함: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

# YoloDetector::detectFarRoadObjects 와 같은 비율
CROP_X, CROP_Y, CROP_W, CROP_H = 0.25, 0.38, 0.50, 0.36


def crop_region(frame):
    """전체 프레임에서 원거리 crop 영역을 잘라냄. C++ 쪽과 같은 반올림을 씀"""
    height, width = frame.shape[:2]
    x = int(round(width * CROP_X))
    y = int(round(height * CROP_Y))
    w = int(round(width * CROP_W))
    h = int(round(height * CROP_H))
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)
    return frame[y:y + h, x:x + w]


def sample_indices(total, count, skip_head):
    """skip_head 이후 구간에서 count 개를 균등 간격으로 고름"""
    usable = total - skip_head
    if usable <= 0:
        return []
    count = min(count, usable)
    step = usable / count
    return [skip_head + int(i * step + step / 2) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="영상 파일들")
    parser.add_argument("--out", default="calib")
    parser.add_argument("--per-video", type=int, default=15)
    parser.add_argument("--skip-head", type=int, default=5)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    full_dir = os.path.join(args.out, "full")
    crop_dir = os.path.join(args.out, "crop")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    full_paths = []
    crop_paths = []
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    for video in args.videos:
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            print(f"[SKIP] 영상을 못 엶: {video}", file=sys.stderr)
            continue
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = os.path.splitext(os.path.basename(video))[0]
        wanted = set(sample_indices(total, args.per_video, args.skip_head))
        if not wanted:
            print(f"[SKIP] 프레임이 너무 적음 ({total}): {video}", file=sys.stderr)
            continue

        index = 0
        saved = 0
        # seek 대신 순서대로 읽음. 압축 영상에서 seek 는 프레임이 어긋날 수 있음
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                name = f"{stem}_{index:06d}.jpg"
                full_path = os.path.join(full_dir, name)
                crop_path = os.path.join(crop_dir, name)
                cv2.imwrite(full_path, frame, jpeg)
                cv2.imwrite(crop_path, crop_region(frame), jpeg)
                full_paths.append(full_path)
                crop_paths.append(crop_path)
                saved += 1
            index += 1
        capture.release()
        print(f"[OK] {video}: {total}프레임 중 {saved}장")

    with open(os.path.join(args.out, "full_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(full_paths) + "\n")
    with open(os.path.join(args.out, "crop_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(crop_paths) + "\n")

    print(f"[DONE] full {len(full_paths)}장, crop {len(crop_paths)}장 → {args.out}/full_list.txt, {args.out}/crop_list.txt")
    if len(full_paths) < 100:
        print("[WARN] calibration 이미지가 100장 미만임. 영상을 더 넣거나 --per-video 를 늘릴 것", file=sys.stderr)


if __name__ == "__main__":
    main()