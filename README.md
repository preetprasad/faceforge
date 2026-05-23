# CITS4402 — Face Detection and Matching

**Student:** Preet Prasad — 23857493
**Unit:** CITS4402 Computer Vision, University of Western Australia

A face detection and identity clustering pipeline with a tkinter GUI.
Single-image mode shows detection, landmarks, and aligned thumbnails; bulk
mode clusters faces across a folder into unique identities.

**For methodology, results, iteration history, and citations see [`Prasad_Report.pdf`](Prasad_Report.pdf).**
This README covers only how to install and run the project.

---

## Setup

### Prerequisites

- Python 3.9 or later (tested on 3.14)
- pip
- A C++ compiler if `dlib` does not have a pre-built wheel for your platform
  (most Linux/macOS users get a wheel automatically; see below if it fails)

### Install dependencies

```bash
pip install -r requirements.txt
```

Verify the critical dependency:

```bash
python -c "import face_recognition; print('face_recognition OK')"
```

If this errors out, the `dlib` wheel was not available for your platform and
must be built from source. Install the prerequisites first, then retry:

```bash
# macOS (Homebrew)
brew install cmake
pip install dlib face-recognition

# Ubuntu / Debian
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
pip install dlib face-recognition
```

If `face_recognition` still cannot be installed, the pipeline runs Task 5
on a NumPy LBP fallback with reduced clustering quality — Tasks 1–4 are
unaffected.

### Model files (already in the repo)

| File | Size | Purpose |
|---|---|---|
| `blaze_face_short_range.tflite` | 230 KB | MediaPipe BlazeFace short-range detector |
| `blaze_face_full_range.tflite`  | 1.1 MB | MediaPipe BlazeFace full-range detector  |
| `face_landmarker.task`           | 3.6 MB | MediaPipe FaceLandmarker (478 landmarks)  |

If any of these are missing the pipeline transparently falls back to OpenCV
Haar cascades (bundled inside `opencv-python`) — no manual download needed.

---

## Run

```bash
python Prasad_Script.py
```

This opens the GUI. Two buttons:

- **Single Image** — pick a `.jpg`/`.jpeg`/`.png`/`.bmp`. Left panel shows
  the original; right panel shows green face boxes, coloured landmark dots
  (red = right eye, green = left eye, blue = nose tip), and aligned 125×125
  thumbnails pasted into the image corners. Status bar reports time and
  face count.

- **Bulk Processing** — pick a folder of images. The pipeline runs on every
  image, computes 128-D face embeddings, clusters them with DBSCAN, and
  saves `Identity_N_face_M.jpg` to `Processed_Images/` next to the chosen
  folder. Status bar reports total images, total faces, and number of
  unique identities found. Any pre-existing files in `Processed_Images/`
  are deleted before the new run starts.

---

## Run the test suite (optional)

```bash
python test_pipeline.py
```

Per-image and per-face report of detection counts, landmark-in-box checks,
alignment success, output sizes, and per-stage timing. Annotated outputs
written to `test_outputs/`.

---

## Submission contents

```
face-detection/
├── Prasad_Script.py               # Pipeline + GUI
├── test_pipeline.py               # Optional test suite
├── requirements.txt
├── README.md                      # This file
├── Prasad_Report.pdf              # Project report
│
├── blaze_face_short_range.tflite  # MediaPipe BlazeFace short-range
├── blaze_face_full_range.tflite   # MediaPipe BlazeFace full-range
├── face_landmarker.task           # MediaPipe FaceLandmarker
│
├── sample_images/                 # 12 provided test images (1.jpg–12.jpg)
└── Processed_Images/              # Created automatically by bulk processing
```

---

## Dependencies (versions tested)

| Package | Version | Used for |
|---|---|---|
| `opencv-python`    | 4.13.0  | Haar cascades, image ops, warpAffine |
| `numpy`            | 2.4.2   | LBP, array math |
| `Pillow`           | 12.1.1  | tkinter image display |
| `scikit-learn`     | 1.8.0   | DBSCAN, silhouette score |
| `face-recognition` | 1.3.0   | dlib ResNet 128-D embeddings |
| `mediapipe`        | 0.10.35 | BlazeFace + FaceLandmarker |
| `tkinter`          | stdlib  | GUI |
