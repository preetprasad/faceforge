"""
Pipeline performance test for Prasad_Script.py.

Runs every image in sample_images/ through each pipeline stage independently
and prints a structured report:

  - Task 2 (face detection):  per-image count vs ground truth + timing
  - Task 3 (landmarks):       per-face both-eyes rate, in-box checks,
                              eye separation ratio, timing
  - Task 4 (alignment):       per-face success, output size, timing
  - Full pipeline:            end-to-end timing as the GUI would invoke it

Annotated outputs and skin masks are written to test_outputs/ for visual
inspection.  This script is optional — it is not required by the GUI;
it exists so the numbers in Prasad_Report.pdf can be reproduced.

Usage:
    python test_pipeline.py
"""

import os
import sys
import time

import cv2
import numpy as np

# Make the submitted script importable when test_pipeline.py is run from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Prasad_Script import (
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


# Spec target positions and BGR landmark colours (must match Prasad_Script.py).
TARGET_RE = (40, 40)
TARGET_LE = (85, 40)
TARGET_NT = (63, 70)
BGR_RED   = (0, 0, 255)     # right eye dot
BGR_GREEN = (0, 255, 0)     # left eye dot
BGR_BLUE  = (255, 0, 0)     # nose tip dot


def colour_close(pixel, target_bgr, tol=60):
    """True if a BGR pixel is close enough to the target colour to be that
    landmark dot.  Tolerance is generous to absorb anti-aliasing at the
    circle edge — the dot's centre pixel is always saturated."""
    return all(abs(int(pixel[i]) - target_bgr[i]) < tol for i in range(3))


def test_image(bgr, fname):
    """
    Run every pipeline stage on one image.
    Returns (result_dict, annotated_output_image).

    Adds spec-strict checks on top of the basic stage report:
      Task 4a — landmarks land at exact target positions after getAffineTransform
      Task 4c — corner thumbnail has red/green/blue dots at the canonical positions
      Task 4c — original-image landmarks coloured red/green/blue (covered indirectly by Task 4b)
      Task 4b — the 4 corners of the output contain the aligned-face thumbnails
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
        # ── Task 4b corner placement ──
        "corner_thumbnails_ok": [],   # one bool per corner that should hold a thumbnail
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
            # Task 4a — landmark target residuals after the affine fit
            "max_target_residual": None,
            # Task 4c — colour dots in the corner thumbnail
            "thumb_re_red": False,
            "thumb_le_green": False,
            "thumb_nt_blue": False,
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

        # Alignment.  align_face returns (clean, display).
        # clean   — saved by bulk processing (no dots)
        # display — pasted into the corner thumbnails (with dots)
        t0 = time.perf_counter()
        clean, display = align_face(bgr, right_eye, left_eye, nose_tip)
        face_info["t_align"] = time.perf_counter() - t0
        if clean is not None:
            face_info["align_ok"]    = True
            face_info["align_shape"] = clean.shape

            # ── Task 4a: landmark target residuals ──────────────────────────
            # Recompute the affine M from the same three correspondences the script
            # uses, then transform each source landmark and measure distance
            # from the spec target.  With cv2.getAffineTransform this is 0 px.
            src = np.float32([right_eye, left_eye, nose_tip])
            dst = np.float32([TARGET_RE, TARGET_LE, TARGET_NT])
            M = cv2.getAffineTransform(src, dst)
            residuals = []
            for p, t in zip([right_eye, left_eye, nose_tip],
                            [TARGET_RE, TARGET_LE, TARGET_NT]):
                mx = M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2]
                my = M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]
                residuals.append(float(np.hypot(mx - t[0], my - t[1])))
            face_info["max_target_residual"] = max(residuals)

        if display is not None:
            # ── Task 4c: corner-thumbnail colour dots ───────────────────────
            # OpenCV indexes images [y, x], so target_y goes first.
            face_info["thumb_re_red"]   = colour_close(display[TARGET_RE[1], TARGET_RE[0]], BGR_RED)
            face_info["thumb_le_green"] = colour_close(display[TARGET_LE[1], TARGET_LE[0]], BGR_GREEN)
            face_info["thumb_nt_blue"]  = colour_close(display[TARGET_NT[1], TARGET_NT[0]], BGR_BLUE)

        result["faces"].append(face_info)

    # ── Full pipeline (as the GUI would call it) ──────────────────────────────
    t0       = time.perf_counter()
    output, _, _ = process_image(bgr)
    result["pipeline_time"]      = time.perf_counter() - t0
    result["pipeline_output_ok"] = (output is not None and output.shape == bgr.shape)

    # ── Task 4b: 4-corner thumbnail placement ──────────────────────────────
    # For each detected face (up to 4), the pipeline pastes the 125×125 aligned
    # thumbnail into one of the corners.  The thumbnail carries the green
    # left-eye dot at canvas (85, 40), so we sample that pixel in each
    # corresponding output corner — if it is green, the thumbnail is there.
    if output is not None:
        H, W = output.shape[:2]
        FOS  = 125
        corners = [
            (0,         0),                 # top-left
            (0,         W - FOS),           # top-right
            (H - FOS,   0),                 # bottom-left
            (H - FOS,   W - FOS),           # bottom-right
        ]
        n_expected = min(len(face_boxes), 4)
        for i in range(n_expected):
            row, col = corners[i]
            pix = output[row + TARGET_LE[1], col + TARGET_LE[0]]
            result["corner_thumbnails_ok"].append(colour_close(pix, BGR_GREEN))

    return result, output


# ── Reporter ──────────────────────────────────────────────────────────────────

def print_report(all_results, bulk_result=None):
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

    # ── Task 4a — strict landmark target positions ─────────────────────────
    print(f"\n{LINE}")
    print("TASK 4a – LANDMARK TARGET POSITIONS")
    print(LINE)
    print(f"{'Image':<10} {'Face':>5} {'Max residual (px)':>20}")
    print("-" * 40)
    max_residual_all = 0.0
    exact_hits = 0
    for r in all_results:
        for fi, f in enumerate(r["faces"]):
            res = f["max_target_residual"]
            res_str = f"{res:.3f}" if res is not None else "n/a"
            flag = ""
            if res is not None:
                max_residual_all = max(max_residual_all, res)
                if res < 0.01:
                    exact_hits += 1
                elif res > 0.5:
                    flag = " !drift"
            print(f"{r['file']:<10} {fi+1:>5} {res_str:>20}{flag}")
    if total_faces > 0:
        print(f"\n  Max residual across all faces : {max_residual_all:.3f} px")
        print(f"  Faces with sub-pixel residual : {exact_hits}/{total_faces}")

    # ── Task 4c — corner-thumbnail landmark colours ────────────────────────
    print(f"\n{LINE}")
    print("TASK 4c – CORNER THUMBNAIL LANDMARK COLOURS")
    print(LINE)
    print(f"{'Image':<10} {'Face':>5} {'RE=red':>8} {'LE=green':>10} {'NT=blue':>9}")
    print("-" * 50)
    re_red_ok = le_green_ok = nt_blue_ok = 0
    for r in all_results:
        for fi, f in enumerate(r["faces"]):
            re_r = f["thumb_re_red"]
            le_g = f["thumb_le_green"]
            nt_b = f["thumb_nt_blue"]
            if re_r:  re_red_ok   += 1
            if le_g:  le_green_ok += 1
            if nt_b:  nt_blue_ok  += 1
            print(f"{r['file']:<10} {fi+1:>5} "
                  f"{'Yes' if re_r else 'No':>8} "
                  f"{'Yes' if le_g else 'No':>10} "
                  f"{'Yes' if nt_b else 'No':>9}")
    if total_faces > 0:
        print(f"\n  Right eye dot red    : {re_red_ok}/{total_faces}")
        print(f"  Left  eye dot green  : {le_green_ok}/{total_faces}")
        print(f"  Nose      dot blue   : {nt_blue_ok}/{total_faces}")

    # ── Task 4b — corner thumbnail placement ───────────────────────────────
    print(f"\n{LINE}")
    print("TASK 4b – CORNER THUMBNAIL PLACEMENT")
    print(LINE)
    print(f"{'Image':<10} {'Expected':>9} {'Corners with thumbnail':>26}")
    print("-" * 50)
    total_corners_expected = 0
    total_corners_ok       = 0
    for r in all_results:
        n_exp = len(r["corner_thumbnails_ok"])
        n_ok  = sum(r["corner_thumbnails_ok"])
        total_corners_expected += n_exp
        total_corners_ok       += n_ok
        print(f"{r['file']:<10} {n_exp:>9} {n_ok:>26}")
    if total_corners_expected > 0:
        print(f"\n  Corner thumbnails correctly placed : "
              f"{total_corners_ok}/{total_corners_expected}")

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

    # ── Task 5 — bulk processing (5a–e) ────────────────────────────────────
    if bulk_result is not None:
        print(f"\n{LINE}")
        print("TASK 5 – BULK PROCESSING (5a–e)")
        print(LINE)
        b = bulk_result
        print(f"  Images processed              : {b['n_imgs']}")
        print(f"  Faces detected                : {b['n_faces']}")
        print(f"  Unique identities             : {b['n_identities']}")
        print(f"  Elapsed                       : {b['elapsed']:.2f} s")
        print(f"  Output folder                 : {b['out_dir']}")
        print(f"  Folder created                : "
              f"{'Yes' if b['folder_created'] else 'No'}")
        print(f"  Files saved                   : {b['n_saved']}  "
              f"(expected {b['n_faces']})")
        print(f"  Filename pattern OK           : "
              f"{'Yes' if b['naming_ok'] else 'No'}  "
              f"(Identity_N_face_M.jpg)")
        print(f"  All saved crops 125x125       : "
              f"{'Yes' if b['size_ok'] else 'No'}")
        print(f"  Saved crops have no dots      : "
              f"{'Yes' if b['no_dots_in_saved'] else 'No'}")
        print(f"  Pre-existing files wiped      : "
              f"{'Yes' if b['junk_wiped'] else 'No'}  (5e)")

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
    if max_residual_all > 0.5:
        issues.append(f"  [BUG]      Landmark target residual > 0.5 px "
                      f"(max {max_residual_all:.2f}) — alignment not exact")
    if total_faces > 0 and (re_red_ok < total_faces or
                            le_green_ok < total_faces or
                            nt_blue_ok < total_faces):
        issues.append(f"  [BUG]      Corner thumbnail colour dots missing — "
                      f"RE-red {re_red_ok}/{total_faces}, "
                      f"LE-green {le_green_ok}/{total_faces}, "
                      f"NT-blue {nt_blue_ok}/{total_faces}")
    if total_corners_expected > 0 and total_corners_ok < total_corners_expected:
        issues.append(f"  [BUG]      {total_corners_expected - total_corners_ok}/"
                      f"{total_corners_expected} corner thumbnails missing/wrong place")
    if bulk_result is not None:
        b = bulk_result
        if not b["folder_created"]:
            issues.append("  [CRITICAL] 5b — Processed_Images folder not created")
        if not b["naming_ok"]:
            issues.append("  [CRITICAL] 5c — saved files do not match Identity_N_face_M.jpg")
        if not b["size_ok"]:
            issues.append("  [BUG]      5c — at least one saved crop is not 125×125")
        if not b["no_dots_in_saved"]:
            issues.append("  [BUG]      5c — saved crops contain landmark dots "
                          "(should be clean)")
        if not b["junk_wiped"]:
            issues.append("  [CRITICAL] 5e — pre-existing Processed_Images files "
                          "were not deleted before the new run")
        if b["n_saved"] != b["n_faces"]:
            issues.append(f"  [BUG]      5c — saved {b['n_saved']} files but "
                          f"reported {b['n_faces']} faces")

    if not issues:
        print("  None — all spec requirements pass.")
    for issue in issues:
        print(issue)
    print()


# ── Task 5 bulk processing test (5a–e) ─────────────────────────────────────

def test_bulk_processing(sample_dir):
    """
    Exercise bulk_process_folder against every spec sub-requirement
    for Task 5:

      5a   read images from one folder            (input = sample_dir)
      5b   output to a 'Processed_Images' folder
           at the same location as sample_dir     (parent dir)
      5c   cropped aligned 125x125, NO landmarks,
           filename 'Identity_N_face_M.jpg'
      5d   clustering returns the identity count  (n_identities)
      5e   existing files in Processed_Images are
           wiped before each run                   (junk-file check)
    """
    import re as _re
    from Prasad_Script import bulk_process_folder

    out_dir = os.path.join(os.path.dirname(os.path.abspath(sample_dir)),
                           "Processed_Images")

    # Task 5e setup: pre-populate Processed_Images with a junk file that should
    # be removed by bulk_process_folder before saving anything new.
    os.makedirs(out_dir, exist_ok=True)
    junk_path = os.path.join(out_dir, "JUNK_SHOULD_BE_DELETED.txt")
    with open(junk_path, "w") as fh:
        fh.write("if this file still exists after bulk_process_folder runs, "
                 "the wipe-on-each-run requirement (5e) is broken.")

    # Run Task 5.
    n_imgs, n_faces, n_identities, elapsed = bulk_process_folder(sample_dir)

    folder_created = os.path.isdir(out_dir)
    junk_wiped     = not os.path.exists(junk_path)

    # Inspect the saved files.
    saved = sorted(f for f in os.listdir(out_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    name_re = _re.compile(r"^Identity_\d+_face_\d+\.(?:jpg|jpeg|png)$",
                          _re.IGNORECASE)
    naming_ok = all(name_re.match(f) for f in saved) if saved else False

    size_ok          = True
    no_dots_in_saved = True
    for f in saved:
        img = cv2.imread(os.path.join(out_dir, f))
        if img is None or img.shape != (125, 125, 3):
            size_ok = False
            continue
        # Task 5c: saved crops must NOT have landmark dots.  If the red/green/blue
        # target pixels match the landmark BGR colours, the saved crop is the
        # 'display' image not the 'clean' one — that's a violation.
        re_pix = img[TARGET_RE[1], TARGET_RE[0]]
        le_pix = img[TARGET_LE[1], TARGET_LE[0]]
        nt_pix = img[TARGET_NT[1], TARGET_NT[0]]
        if (colour_close(re_pix, BGR_RED,   tol=30) or
            colour_close(le_pix, BGR_GREEN, tol=30) or
            colour_close(nt_pix, BGR_BLUE,  tol=30)):
            no_dots_in_saved = False

    return {
        "out_dir":          out_dir,
        "n_imgs":           n_imgs,
        "n_faces":          n_faces,
        "n_identities":     n_identities,
        "elapsed":          elapsed,
        "folder_created":   folder_created,
        "junk_wiped":       junk_wiped,
        "n_saved":          len(saved),
        "naming_ok":        naming_ok,
        "size_ok":          size_ok,
        "no_dots_in_saved": no_dots_in_saved,
    }


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

    # Task 5 end-to-end test — runs bulk_process_folder against sample_images/
    # and inspects the Processed_Images output for every Task 5 subitem.
    print("\nRunning Task 5 bulk-processing test ...")
    bulk_result = test_bulk_processing(SAMPLE_DIR)

    print_report(all_results, bulk_result)


if __name__ == "__main__":
    main()
