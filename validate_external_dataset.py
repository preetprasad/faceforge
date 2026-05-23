import argparse
import csv
import os
import random
import time

import cv2
from PIL import Image, ImageDraw

from Prasad_Script import process_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root):
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name.lower())[1] in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def make_contact_sheet(rows, output_path, thumb_w=220, thumb_h=165):
    if not rows:
        return

    cells = []
    for rel_path, original, processed, n_faces in rows:
        cell = Image.new("RGB", (thumb_w * 2, thumb_h + 24), "white")
        draw = ImageDraw.Draw(cell)
        draw.text((5, 4), f"{rel_path} | faces={n_faces}", fill=(0, 0, 0))

        for x, bgr in ((0, original), (thumb_w, processed)):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (thumb_w, thumb_h),
                             interpolation=cv2.INTER_AREA)
            cell.paste(Image.fromarray(rgb), (x, 24))
        cells.append(cell)

    cols = 2
    out_w = cols * thumb_w * 2
    out_h = ((len(cells) + cols - 1) // cols) * (thumb_h + 24)
    sheet = Image.new("RGB", (out_w, out_h), "white")
    for i, cell in enumerate(cells):
        x = (i % cols) * thumb_w * 2
        y = (i // cols) * (thumb_h + 24)
        sheet.paste(cell, (x, y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, quality=92)


def main():
    parser = argparse.ArgumentParser(
        description="Run Prasad_Script.py on an external image dataset.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--seed", type=int, default=4402)
    parser.add_argument("--output-dir", default="external_eval_outputs")
    args = parser.parse_args()

    image_paths = find_images(args.dataset_dir)
    if not image_paths:
        raise SystemExit(f"No images found under {args.dataset_dir}")

    rng = random.Random(args.seed)
    if len(image_paths) > args.max_images:
        image_paths = rng.sample(image_paths, args.max_images)
    image_paths = sorted(image_paths)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    contact_path = os.path.join(args.output_dir, "contact_sheet.jpg")

    rows_for_sheet = []
    total_faces = 0
    zero_face_images = 0
    t0 = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "faces", "time_ms"])
        for path in image_paths:
            bgr = cv2.imread(path)
            if bgr is None:
                continue

            start = time.perf_counter()
            processed, n_faces, _ = process_image(bgr)
            elapsed_ms = (time.perf_counter() - start) * 1000

            rel = os.path.relpath(path, args.dataset_dir)
            writer.writerow([rel, n_faces, f"{elapsed_ms:.1f}"])
            total_faces += n_faces
            zero_face_images += int(n_faces == 0)

            if len(rows_for_sheet) < 24:
                rows_for_sheet.append((rel, bgr, processed, n_faces))

    make_contact_sheet(rows_for_sheet, contact_path)

    elapsed = time.time() - t0
    print(f"Images processed: {len(image_paths)}")
    print(f"Faces detected: {total_faces}")
    print(f"Images with zero faces: {zero_face_images}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"CSV: {csv_path}")
    print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
