import argparse
import csv
import json
import os
import random
import time

import cv2
from PIL import Image, ImageDraw

from Prasad_Script import detect_faces


def parse_boxes(value):
    if not value:
        return []
    return [tuple(map(int, box)) for box in json.loads(value)]


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def match_boxes(gt_boxes, pred_boxes, iou_threshold):
    pairs = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            pairs.append((box_iou(gt, pred), gi, pi))
    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    for iou, gi, pi in pairs:
        if iou < iou_threshold:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)

    tp = len(used_gt)
    fp = len(pred_boxes) - len(used_pred)
    fn = len(gt_boxes) - len(used_gt)
    return tp, fp, fn


def read_labels(labels_csv):
    with open(labels_csv, newline="") as f:
        return list(csv.DictReader(f))


def make_fail_sheet(failures, images_dir, output_path):
    if not failures:
        return

    thumb_w, thumb_h = 260, 195
    cells = []
    for row in failures[:24]:
        image_path = os.path.join(images_dir, row["image"])
        bgr = cv2.imread(image_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((thumb_w, thumb_h))
        draw = ImageDraw.Draw(im)
        sx = thumb_w / bgr.shape[1]
        sy = thumb_h / bgr.shape[0]

        for x, y, w, h in parse_boxes(row.get("boxes_json", "")):
            draw.rectangle(
                [x * sx, y * sy, (x + w) * sx, (y + h) * sy],
                outline=(40, 220, 40),
                width=2,
            )
        for x, y, w, h in parse_boxes(row.get("pred_boxes_json", "")):
            draw.rectangle(
                [x * sx, y * sy, (x + w) * sx, (y + h) * sy],
                outline=(255, 40, 40),
                width=2,
            )
        draw.text((5, 5), f"gt={row['expected_faces']} pred={row['pred_faces']}",
                  fill=(255, 255, 255))
        cells.append(im)

    cols = 3
    sheet = Image.new(
        "RGB",
        (cols * thumb_w, ((len(cells) + cols - 1) // cols) * thumb_h),
        "white",
    )
    for idx, cell in enumerate(cells):
        sheet.paste(cell, ((idx % cols) * thumb_w, (idx // cols) * thumb_h))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Prasad_Script.py Task 2 face detection on a labeled folder."
    )
    parser.add_argument("labels_csv")
    parser.add_argument("--images-dir")
    parser.add_argument("--output-dir", default="external_eval_outputs/task2")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=4402)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    args = parser.parse_args()

    labels_csv = os.path.abspath(args.labels_csv)
    base_dir = os.path.dirname(labels_csv)
    images_dir = args.images_dir or os.path.join(base_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = base_dir

    rows = read_labels(labels_csv)
    rng = random.Random(args.seed)
    if args.max_images and len(rows) > args.max_images:
        rows = rng.sample(rows, args.max_images)
    rows = sorted(rows, key=lambda row: row["image"])

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "task2_results.csv")
    failures = []
    total_tp = total_fp = total_fn = 0
    count_correct = 0
    processed = 0
    started = time.time()

    with open(results_path, "w", newline="") as f:
        fieldnames = [
            "image", "expected_faces", "pred_faces", "tp", "fp", "fn",
            "pred_boxes_json", "boxes_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            image_path = os.path.join(images_dir, row["image"])
            bgr = cv2.imread(image_path)
            if bgr is None:
                continue

            gt_count = int(row["expected_faces"])
            gt_boxes = parse_boxes(row.get("boxes_json", ""))
            pred_boxes = [tuple(map(int, box)) for box in detect_faces(bgr)]

            if gt_boxes:
                tp, fp, fn = match_boxes(gt_boxes, pred_boxes, args.iou_threshold)
            else:
                tp = min(gt_count, len(pred_boxes))
                fp = max(0, len(pred_boxes) - gt_count)
                fn = max(0, gt_count - len(pred_boxes))

            result = {
                "image": row["image"],
                "expected_faces": str(gt_count),
                "pred_faces": str(len(pred_boxes)),
                "tp": str(tp),
                "fp": str(fp),
                "fn": str(fn),
                "pred_boxes_json": json.dumps(pred_boxes, separators=(",", ":")),
                "boxes_json": row.get("boxes_json", ""),
            }
            writer.writerow(result)

            processed += 1
            total_tp += tp
            total_fp += fp
            total_fn += fn
            if len(pred_boxes) == gt_count and fp == 0 and fn == 0:
                count_correct += 1
            else:
                failures.append(result)

    fail_sheet = os.path.join(args.output_dir, "task2_failures.jpg")
    if failures:
        make_fail_sheet(failures, images_dir, fail_sheet)
    elif os.path.exists(fail_sheet):
        os.remove(fail_sheet)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    print(f"Images evaluated: {processed}")
    print(f"Count-correct images: {count_correct}/{processed}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f}")
    print(f"Elapsed seconds: {time.time() - started:.2f}")
    print(f"CSV: {results_path}")
    if failures:
        print(f"Failure sheet: {fail_sheet}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()
