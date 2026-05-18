# CITS4402 Computer Vision — Face Detection and Matching

**Student:** Preet Prasad — 23857493  
**Unit:** CITS4402 Computer Vision, University of Western Australia  

---

## Overview

A complete face detection and identity clustering pipeline built on OpenCV, MediaPipe, and scikit-learn. The system processes colour images containing 1–4 faces, detects each face, localises three facial landmarks, aligns each face to a canonical 125×125 crop, and clusters all faces across a folder into unique identities. A tkinter GUI presents the input and processed output side-by-side.

---

## Pipeline

```
┌─ Task 1: GUI (tkinter)
│     Two Canvas panels: Input Image | Processed Image
│     Button A — Single Image: load → process → display
│     Button B — Bulk Processing: folder → cluster → save Processed_Images/
│     Status label: processing time, face count, identity count
│
Image (loaded via Button A or B)
  │
  ├─ Task 2: Face Detection
  │     MediaPipe BlazeFace (primary, ~15 ms/image)
  │     └─ Haar cascade fallback (frontalface_default + alt2 + profile)
  │     └─ Three-rule skin filter (YCrCb + HSV + RGB majority vote)
  │
  ├─ Task 3: Landmark Detection  (per detected face)
  │     MediaPipe FaceLandmarker (iris landmarks 468/473, nose tip 4)
  │     └─ Hough circle transform fallback (Lab 04 edge-overlap scoring)
  │     └─ Haar eye cascade fallback → geometric estimate
  │
  ├─ Task 4: Face Alignment
  │     cv2.estimateAffinePartial2D  →  cv2.warpAffine  →  125×125 crop
  │     Landmarks mapped to: right eye (40,40), left eye (85,40), nose (63,70)
  │
  └─ Task 5: Identity Clustering  (bulk mode only)
        dlib ResNet 128-D embeddings via face_recognition (primary)
        └─ NumPy LBP 5×5 grid fallback (no pretrained model required)
        DBSCAN eps=0.38, min_samples=1
        → Identity_N_face_M.jpg saved to Processed_Images/
```

---

## Results on Sample Images

| Metric | Result |
|---|---|
| Face detection | 12/12 images, all counts correct |
| Both eyes detected | 26/26 (100%) |
| All landmarks inside bounding box | 100% |
| Alignment success | 26/26, all exactly 125×125 px |
| Avg pipeline time | ~35 ms/image |
| DBSCAN clusters (26 faces) | 6 clusters, silhouette = 0.214 |

---

## Project Structure

```
face-detection/
├── draft.py                      # Main submission — all pipeline code + GUI
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── Report.docx                   # Project report
│
├── blaze_face_short_range.tflite # MediaPipe BlazeFace short-range model (230 KB)
├── blaze_face_full_range.tflite  # MediaPipe BlazeFace full-range model (1.1 MB)
├── face_landmarker.task          # MediaPipe FaceLandmarker model (3.6 MB)
│
├── sample_images/                # 12 provided test images (1.jpg – 12.jpg)
├── Processed_Images/             # Created automatically by bulk processing
│                                 # Contains Identity_N_face_M.jpg outputs
│
├── test_baseline.py              # Automated test suite (detection / landmark / alignment)
└── make_report.py                # Script that regenerates Report.docx
```

> **Note:** `Processed_Images/` is deleted and recreated each time bulk processing runs. It is listed in `.gitignore` and is not part of the submission.

---

## Setup

### Prerequisites

- Python 3.9 or later (tested on 3.14)
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

`face-recognition` depends on `dlib`, which requires a C++ compiler. On most systems `pip` will install a pre-built wheel. If it fails, install dlib first:

```bash
# macOS (Homebrew)
brew install cmake
pip install dlib
pip install face-recognition

# Ubuntu / Debian
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
pip install dlib
pip install face-recognition
```

All other packages (`opencv-python`, `mediapipe`, `scikit-learn`, `Pillow`, `numpy`) install without additional system dependencies.

### Model files

The three MediaPipe model files are included in the repository root:

| File | Size | Purpose |
|---|---|---|
| `blaze_face_short_range.tflite` | 230 KB | BlazeFace short-range face detector |
| `blaze_face_full_range.tflite` | 1.1 MB | BlazeFace full-range face detector |
| `face_landmarker.task` | 3.6 MB | 478-landmark face mesh + iris |

If the `.tflite`/`.task` files are missing, the pipeline automatically falls back to the OpenCV Haar cascades bundled inside `opencv-python` — no manual download is required.

---

## Running the Application

```bash
python draft.py
```

This opens the GUI window. Two buttons are available:

### Single Image

1. Click **Single Image**.
2. Select any `.jpg`, `.jpeg`, `.png`, or `.bmp` file.
3. The left panel shows the original image; the right panel shows:
   - Green bounding boxes around each detected face.
   - Red dot = subject's right eye, green dot = subject's left eye, blue dot = nose tip.
   - Up to four 125×125 aligned face thumbnails pasted into the image corners.
4. The status bar reports processing time and face count.

### Bulk Processing

1. Click **Bulk Processing**.
2. Select a folder containing image files.
3. The pipeline runs on every image in the folder, computes 128-D face embeddings, and clusters them with DBSCAN.
4. Results are saved to `Processed_Images/` (created next to the selected folder):
   - `Identity_N_face_M.jpg` — the M-th detected face, assigned to cluster N.
5. The status bar reports total images, total faces, and number of unique identities found.

---

## Technical Details

### Task 2 — Face Detection

The primary detector is **MediaPipe BlazeFace** (both the short-range and full-range TFLite models). Both models run on CPU; results from each are merged and deduplicated with IoU-based non-maximum suppression (threshold 0.30). Each surviving box is expanded upward by 30% to capture the forehead when the detector anchors on the lower face.

A **three-rule majority-vote skin filter** (Lab 05 technique) validates each candidate box. A pixel is classified as skin when at least two of three colour-space rules agree:
- YCrCb: Cr ∈ [133, 183], Cb ∈ [75, 135]
- HSV: H ∈ [0, 28] ∪ [160, 179] (hue wrap-around for reddish skin)
- Normalised RGB: R dominates (R > G − 5, R > B, R > 70)

Boxes with less than 2% skin pixels are discarded. The final result is capped at 4 (the specification's maximum) by ranking candidates on confidence × skin-coverage score.

When MediaPipe is unavailable the fallback uses `haarcascade_frontalface_default` and `haarcascade_frontalface_alt2` with the same NMS, expansion, and skin filter. `haarcascade_profileface` is used as a last resort when the frontal cascade finds nothing.

### Task 3 — Landmark Detection

The primary method is **MediaPipe FaceLandmarker**, which produces 478 3-D landmarks. Iris centres are taken from landmarks 468 (subject's right) and 473 (subject's left); nose tip from landmark 4. Coordinates are projected back to pixel space using the bounding box dimensions.

The Hough-circle fallback (Lab 04) applies `cv2.HoughCircles` to the upper 58% of the face ROI. Each circle candidate is scored by edge-overlap fraction (Canny edge pixels on the circle perimeter, same technique as the Lab 04 challenge) plus a vertical-position score. Successive fallback layers activate for any still-missing landmark:
1. `haarcascade_eye_tree_eyeglasses` on the full upper ROI
2. `haarcascade_righteye_2splits` / `haarcascade_lefteye_2splits` on each face half
3. Proportional geometric estimate as the final fallback

The nose tip is located as the intensity minimum of a Gaussian-blurred patch in the central 30–70% × 50–72% region of the face, exploiting the shadow beneath the nose tip.

A sanity check rejects eye pairs with separation ratio outside [0.18, 0.65] of face width, replacing them with geometric estimates.

### Task 4 — Face Alignment

`cv2.estimateAffinePartial2D` computes a **similarity transform** (rotation + uniform scale + translation) mapping the three detected landmarks to fixed target positions inside a 125×125 canvas. A single `cv2.warpAffine` call simultaneously aligns, crops, and resizes. `BORDER_REPLICATE` fills any out-of-bounds pixels, preventing black borders when faces are near the image edge.

Target positions:
- Subject's right eye → (40, 40)
- Subject's left eye → (85, 40)
- Nose tip → (63, 70)

### Task 5 — Identity Clustering

**Feature extraction:** 128-D L2-normalised embeddings from the dlib ResNet face recognition model (via `face_recognition`). The specification explicitly permits this: *"You are allowed to use a pre-trained model for this."* If `face_recognition` is unavailable, a pure-NumPy LBP descriptor (5×5 spatial grid, 6400-D) is used as a fallback.

**Clustering:** DBSCAN with `eps=0.38`, `min_samples=1`, Euclidean distance on L2-normalised embeddings. `eps=0.38` was selected by sweeping eps from 0.35 to 0.45 and choosing the value with the highest silhouette score (0.214 at eps=0.38). `min_samples=1` ensures every face forms at least one cluster — no face is discarded as noise. A post-processing step merges singleton clusters whose nearest neighbour is within the threshold and does not share a source image.

---

## Running the Test Suite

```bash
python test_baseline.py
```

Runs every sample image through each pipeline stage and prints a structured report covering detection counts (vs ground truth), per-face landmark quality (both-eyes rate, in-box checks, eye-separation ratio), alignment success, output sizes, and per-stage timing.

Annotated outputs are saved to `test_outputs/` for visual inspection.

---

## Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| `opencv-python` | 4.13.0 | Haar cascades, image processing, warpAffine |
| `numpy` | 2.4.2 | LBP computation, array operations |
| `Pillow` | 12.1.1 | tkinter image display |
| `scikit-learn` | 1.8.0 | DBSCAN clustering, silhouette score |
| `face-recognition` | 1.3.0 | dlib ResNet 128-D face embeddings |
| `mediapipe` | 0.10.35 | BlazeFace detection, FaceLandmarker |
| `tkinter` | stdlib | GUI |
