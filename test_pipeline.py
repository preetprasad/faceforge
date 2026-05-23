"""
Pipeline performance test for draft.py.

Runs every image in sample_images/ through each pipeline stage independently
and prints a structured report:

  - Task 2 (face detection):  per-image count vs ground truth + timing
  - Task 3 (landmarks):       per-face both-eyes rate, in-box checks,
                              eye separation ratio, timing
  - Task 4 (alignment):       per-face success, output size, timing
  - Full pipeline:            end-to-end timing as the GUI would invoke it

Annotated outputs and skin masks are written to test_outputs/ for visual
inspection.  This script is optional — it is not required by the GUI;
it exists so the numbers in Report.pdf can be reproduced.

Usage:
    python test_pipeline.py
"""

import os
import sys
import time

import cv2
import numpy as np

# Make draft.py importable when this script is run from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft import (
    detect_faces,
    detect_landmarks,
    align_face,
    process_image,
    skin_segmentation_mask,
)

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")

# Ground truth: expected number of faces per image (verified by visual inspection
# of the 12 sample images).  Mismatches are flagged in the Task 2 report.
GROUND_TRUTH = {
    "1.jpg":  2,
    "2.jpg":  2,
    "3.jpg":  1,
    "4.jpg":  1,
    "5.jpg":  2,
    "6.jpg":  2,
    "7.jpg":  3,
    "8.jpg":  2,
    "9.jpg":  2,
    "10.jpg": 3,
    "11.jpg": 4,
    "12.jpg": 2,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def point_in_box(pt, box):
    """Check whether a point (x,y) lies inside a face box (x,y,w,h)."""
    if pt is None:
        return False
    px, py = pt
    bx, by, bw, bh = box
    return bx <= px <= bx + bw and by <= py <= by + bh


def eye_separation_ratio(left_eye, right_eye, face_w):
    """
    Eye separation relative to face width.
    Typical human value: 0.25 – 0.55.
    Returns None if either eye is missing.
    """
    if left_eye is None or right_eye is None:
        return None
    sep = abs(right_eye[0] - left_eye[0])
    return sep / max(face_w, 1)


def test_image(bgr, fname):
    """
    Run every pipeline stage on one image.
    Returns a dict of measurements.
    """
    result = {
        "file": fname,
        "img_shape": bgr.shape,
        # ── face detection ──
        "n_faces_detected": 0,
        "face_detection_time": 0.0,
        # ── per-face breakdown ──
        "faces": [],          # list of per-face dicts
        # ── full pipeline ──
        "pipeline_time": 0.0,
        "pipeline_output_ok": False,
    }

    # ── Task 2: Face Detection ────────────────────────────────────────────────
    t0 = time.perf_counter()
    face_boxes = detect_faces(bgr)
    result["face_detection_time"] = time.perf_counter() - t0
    result["n_faces_detected"] = len(face_boxes)

    # ── Task 3 + 4: Landmarks & Alignment per face ────────────────────────────
    for idx, box in enumerate(face_boxes):
        fx, fy, fw, fh = box
        face_info = {
            "box": box,
            # landmarks
            "t_landmark": 0.0,
            "left_eye": None,
            "right_eye": None,
            "nose_tip": None,
            "both_eyes_detected": False,
            "left_eye_in_box": False,
            "right_eye_in_box": False,
            "nose_in_box": False,
            "eye_sep_ratio": None,
            # alignment
            "t_align": 0.0,
            "align_ok": False,
            "align_shape": None,
        }

        # Landmark detection (returns right_eye, left_eye, nose_tip)
        t0 = time.perf_counter()
        right_eye, left_eye, nose_tip = detect_landmarks(bgr, box)
        face_info["t_landmark"] = time.perf_counter() - t0
        face_info["left_eye"]  = left_eye
        face_info["right_eye"] = right_eye
        face_info["nose_tip"]  = nose_tip
        face_info["both_eyes_detected"] = (left_eye is not None and right_eye is not None)

        # Sanity checks: are landmarks inside the face box?
        face_info["left_eye_in_box"]  = point_in_box(left_eye,  box)
        face_info["right_eye_in_box"] = point_in_box(right_eye, box)
        face_info["nose_in_box"]      = point_in_box(nose_tip,  box)
        face_info["eye_sep_ratio"]    = eye_separation_ratio(left_eye, right_eye, fw)

        # Alignment.  draft.align_face returns (clean, display); only the
        # clean crop (no landmark dots) is needed to check the alignment.
        t0 = time.perf_counter()
        clean, _ = align_face(bgr, right_eye, left_eye, nose_tip)
        face_info["t_align"] = time.perf_counter() - t0
        if clean is not None:
            face_info["align_ok"]    = True
            face_info["align_shape"] = clean.shape

        result["faces"].append(face_info)

    # ── Full pipeline (as GUI would call it) ──────────────────────────────────
    t0     = time.perf_counter()
    ret    = process_image(bgr)
    output = ret[0]
    n      = ret[1]
    result["pipeline_time"]      = time.perf_counter() - t0
    result["pipeline_output_ok"] = (output is not None and output.shape == bgr.shape)

    return result, output


# ── Reporter ──────────────────────────────────────────────────────────────────

def print_report(all_results):
    LINE = "=" * 78

    print(f"\n{LINE}")
    print("TASK 2 – FACE DETECTION")
    print(LINE)
    header = f"{'Image':<10} {'GT':>4} {'Det':>4} {'Match':>6} {'Time(ms)':>10}"
    print(header)
    print("-" * len(header))

    detection_times = []
    detection_mismatches = 0
    zero_detections = 0

    for r in all_results:
        gt = GROUND_TRUTH.get(r["file"])
        n  = r["n_faces_detected"]
        ms = r["face_detection_time"] * 1000
        detection_times.append(ms)

        if n == 0:
            zero_detections += 1

        if gt is not None:
            match = "OK" if n == gt else f"FAIL({gt} exp)"
            if n != gt:
                detection_mismatches += 1
        else:
            match = "?"

        flag = " !!MISS" if n == 0 else ""
        print(f"{r['file']:<10} {str(gt) if gt is not None else '?':>4} "
              f"{n:>4} {match:>6} {ms:>9.1f}ms{flag}")

    print(f"\n  Images with 0 detections : {zero_detections} / {len(all_results)}")
    if detection_mismatches > 0:
        print(f"  GT mismatches            : {detection_mismatches}")
    print(f"  Avg detection time       : {sum(detection_times)/len(detection_times):.1f} ms")

    # ── Task 3 ────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("TASK 3 – LANDMARK DETECTION  (per detected face)")
    print(LINE)
    print(f"{'Image':<10} {'Face':>5} {'BothEyes':>9} {'LE-inBox':>9} "
          f"{'RE-inBox':>9} {'Nose-inBox':>11} {'EyeSepRatio':>13} {'Time(ms)':>10}")
    print("-" * 80)

    total_faces = 0
    eyes_ok     = 0
    le_in_box   = 0
    re_in_box   = 0
    nose_in_box = 0
    bad_sep     = 0

    for r in all_results:
        for fi, f in enumerate(r["faces"]):
            total_faces += 1
            both   = f["both_eyes_detected"]
            le_ok  = f["left_eye_in_box"]
            re_ok  = f["right_eye_in_box"]
            ni_ok  = f["nose_in_box"]
            eye_sep = f["eye_sep_ratio"]
            ms      = f["t_landmark"] * 1000

            if both:
                eyes_ok += 1
            if le_ok:
                le_in_box += 1
            if re_ok:
                re_in_box += 1
            if ni_ok:
                nose_in_box += 1

            sep_ok = (eye_sep is not None and 0.15 <= eye_sep <= 0.70)
            if eye_sep is not None and not sep_ok:
                bad_sep += 1

            sep_str = f"{eye_sep:.2f}" if eye_sep is not None else "N/A"
            flags = ""
            if not both:
                flags += " !eyes"
            if eye_sep is not None and not sep_ok:
                flags += " !sep"

            print(f"{r['file']:<10} {fi+1:>5} "
                  f"{'Yes' if both else 'No':>9} "
                  f"{'Yes' if le_ok else 'No':>9} "
                  f"{'Yes' if re_ok else 'No':>9} "
                  f"{'Yes' if ni_ok else 'No':>11} "
                  f"{sep_str:>13} "
                  f"{ms:>9.1f}ms{flags}")

    if total_faces > 0:
        print(f"\n  Total faces tested         : {total_faces}")
        print(f"  Both eyes detected         : {eyes_ok}/{total_faces}  ({100*eyes_ok/total_faces:.0f}%)")
        print(f"  Left eye inside box        : {le_in_box}/{total_faces}  ({100*le_in_box/total_faces:.0f}%)")
        print(f"  Right eye inside box       : {re_in_box}/{total_faces}  ({100*re_in_box/total_faces:.0f}%)")
        print(f"  Nose inside box            : {nose_in_box}/{total_faces}  ({100*nose_in_box/total_faces:.0f}%)")
        print(f"  Suspicious eye separation  : {bad_sep}/{total_faces}")
    else:
        print("  No faces detected — cannot evaluate landmarks.")

    # ── Task 4 ────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("TASK 4 – ALIGNMENT")
    print(LINE)
    print(f"{'Image':<10} {'Face':>5} {'AlignOK':>8} {'Shape':>15} {'Time(ms)':>10}")
    print("-" * 55)

    align_ok_count = 0
    wrong_size     = 0

    for r in all_results:
        for fi, f in enumerate(r["faces"]):
            ok    = f["align_ok"]
            shape = f["align_shape"]
            ms    = f["t_align"] * 1000

            if ok:
                align_ok_count += 1
                if shape != (125, 125, 3):
                    wrong_size += 1

            shape_str = str(shape) if shape else "None"
            flags = " !size" if (shape and shape != (125, 125, 3)) else ""
            flags += " !fail" if not ok else ""
            print(f"{r['file']:<10} {fi+1:>5} "
                  f"{'Yes' if ok else 'No':>8} "
                  f"{shape_str:>15} "
                  f"{ms:>9.1f}ms{flags}")

    if total_faces > 0:
        print(f"\n  Alignment success          : {align_ok_count}/{total_faces}  ({100*align_ok_count/total_faces:.0f}%)")
        print(f"  Wrong output size          : {wrong_size}")

    # ── Pipeline timing ───────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("FULL PIPELINE TIMING  (as called by GUI)")
    print(LINE)
    print(f"{'Image':<10} {'Pipeline(ms)':>14} {'OutputOK':>10}")
    print("-" * 38)

    times = []
    for r in all_results:
        ms = r["pipeline_time"] * 1000
        times.append(ms)
        ok = r["pipeline_output_ok"]
        print(f"{r['file']:<10} {ms:>13.1f}ms {'Yes' if ok else 'No':>10}")

    print(f"\n  Avg pipeline time          : {sum(times)/len(times):.1f} ms")
    print(f"  Min / Max                  : {min(times):.1f} ms / {max(times):.1f} ms")

    # ── Overall verdict ───────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("OVERALL ISSUES TO FIX")
    print(LINE)
    issues = []
    if zero_detections:
        issues.append(f"  [CRITICAL] {zero_detections} image(s) got 0 face detections")
    if total_faces > 0 and eyes_ok < total_faces:
        issues.append(f"  [CRITICAL] Eyes not detected for {total_faces - eyes_ok}/{total_faces} faces "
                      f"— landmark quality is low")
    if total_faces > 0 and align_ok_count < total_faces:
        issues.append(f"  [CRITICAL] Alignment failed for {total_faces - align_ok_count}/{total_faces} faces")
    if wrong_size:
        issues.append(f"  [BUG]      {wrong_size} aligned face(s) are not 125×125")
    if bad_sep:
        issues.append(f"  [WARNING]  {bad_sep} face(s) have suspicious eye separation ratio")

    for issue in issues:
        print(issue)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_files = sorted(
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )

    if not img_files:
        print(f"No images found in {SAMPLE_DIR}")
        sys.exit(1)

    print(f"Running pipeline tests on {len(img_files)} images from {SAMPLE_DIR}")
    print(f"Annotated outputs → {OUTPUT_DIR}/")

    all_results = []
    for fname in img_files:
        path = os.path.join(SAMPLE_DIR, fname)
        bgr  = cv2.imread(path)
        if bgr is None:
            print(f"  [SKIP] Could not read {fname}")
            continue

        result, output = test_image(bgr, fname)
        all_results.append(result)

        # Save annotated output for visual inspection
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"out_{fname}"), output)

        # Also save the skin mask for debugging
        mask = skin_segmentation_mask(bgr)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"skin_{fname}"), mask)

    print_report(all_results)


if __name__ == "__main__":
    main()
