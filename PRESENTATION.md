# CITS4402 Face Detection & Matching — Presentation Guide

**Preet Prasad — 23857493**

---

## Quick Overview

I built a face detection and identity clustering pipeline that takes a colour image (640×480, 1–4 faces), detects every face, finds three landmarks per face (right eye, left eye, nose tip), aligns each face to a 125×125 canonical crop, and in bulk mode clusters all faces across a folder into unique identities. It runs a tkinter GUI with two buttons exactly as specified.

---

## Task 1 — GUI

### What I built
- Two `tk.Canvas` panels inside `tk.LabelFrame` containers labelled **"Input Image"** and **"Processed Image"**
- A status label below the panels that shows processing time and face count
- Two buttons: **Single Image** and **Bulk Processing**
- Layout matches the Lab 03/04 canvas-based starter code style

### How it works
- **Single Image**: opens a file dialog → displays original in left panel → runs the full pipeline → displays annotated result in right panel → updates status label
- **Bulk Processing**: opens a folder dialog → iterates every image → runs pipeline → computes embeddings → clusters with DBSCAN → saves `Identity_N_face_M.jpg` to `Processed_Images/` next to the source folder → reports total images, faces, and identities

### Why Canvas not Label
`tk.Label` doesn't scale images to fit the panel — large images overflow the window. `tk.Canvas` with a fixed size and centred image keeps the GUI stable for any input size.

### Cross-questions
**Q: Why does the GUI not resize?**  
`master.resizable(False, False)` keeps the layout fixed. The panels are 500×500 px; images are downscaled to fit without upscaling.

**Q: What happens if the folder has no images?**  
The status label shows "0 images processed" and no `Processed_Images/` folder is created.

**Q: How is `Processed_Images/` handled between runs?**  
`shutil.rmtree` deletes the entire folder before each bulk run, then `os.makedirs` recreates it — exactly as required by the spec.

---

## Task 2 — Face Detection

### Pipeline (in order)

```
1. Compute skin mask (used throughout for scoring)
2. Run MediaPipe BlazeFace (short-range + full-range models)
3. If MediaPipe unavailable → Haar cascade fallback
4. If still nothing → profile cascade last resort
5. IoU-based NMS to deduplicate
6. Cap at 4 (spec guarantees max 4 faces)
```

### Step 1 — Skin-colour segmentation (mandatory)

I implemented a **three-rule majority vote**. A pixel is skin when **at least 2 of 3 rules agree**:

| Rule | Thresholds | Why |
|---|---|---|
| YCrCb | Cr ∈ [133,183], Cb ∈ [75,135] | Luminance-independent; from lab literature |
| HSV | H ∈ [0,28] ∪ [160,179] | Lab 05 finding: "skin has the same Hue regardless of darkness"; union handles red hue wrap-around |
| Normalised RGB | R > G−5, R > B, R > 70 | Luminance-robust; red channel dominates warm skin |

**Why majority vote, not intersection?**  
A simple intersection (YCrCb AND HSV) is too strict — under unusual lighting or darker skin, one rule fires but not the other, causing valid skin pixels to be rejected. Majority vote tolerates one failure while still rejecting backgrounds that fail all three.

**How it assists detection:**  
Candidate boxes are scored with a skin-coverage ratio. Boxes below 2% skin coverage are discarded. High skin coverage also bypasses the Haar re-verification step for large boxes.

**Morphological cleanup:**  
`cv2.morphologyEx` OPEN (removes noise) + CLOSE (fills gaps) with a 5×5 ellipse kernel.

### Step 2 — MediaPipe BlazeFace (primary detector)

Two TFLite models ship in the project folder:
- `blaze_face_short_range.tflite` — 230 KB, optimised for faces 20–300 cm from camera
- `blaze_face_full_range.tflite` — 1.1 MB, broader range

Both run on every image (original + horizontally flipped). Results are merged. Flipping catches right-facing profiles that the detector is weaker on.

**Confidence threshold: 0.28** — deliberately low to maximise recall; downstream skin filter and NMS handle false positives.

**Box expansion:** each detected box is expanded 20% up, 10% left/right, 15% down to capture the full head (detectors often fire on the mid-face, missing the forehead).

**Scoring:** `score = model_confidence + min(skin_coverage, 0.45)` — model confidence is the primary signal, skin coverage is a tiebreaker.

### Step 3 — Haar cascade fallback

Used **only when MediaPipe finds nothing** (e.g., model files missing). Two cascades:
- `haarcascade_frontalface_default.xml` — primary
- `haarcascade_frontalface_alt2.xml` — catches faces tilted ±15°

Both run with `scaleFactor=1.05`, `minNeighbors=4`, `minSize=(36,36)`. A lenient second pass (`minNeighbors=2`) fires if the first finds nothing.

**ROI re-verification:** each Haar candidate box is resized to 120×120 and the cascade is re-run on its interior. This rejects false positives (flower vases, shoulders) that pass the skin filter but don't look like faces under the cascade.

### Step 4 — NMS

Two NMS implementations:
- `_nms_scored_boxes` — for MediaPipe candidates; sorts by score, IoU threshold 0.28, containment > 72%
- `_nms_boxes` — for Haar candidates; sorts by area, IoU threshold 0.30, containment > 65%

**Cap at 4:** spec guarantees at most 4 faces. After NMS, take top 4 by score.

### Why not dlib face detection?

dlib's HOG face detector has lower accuracy on non-frontal faces compared to BlazeFace. dlib's CNN face detector (MMOD) requires a separate model file download and is slower on CPU. MediaPipe BlazeFace achieves better accuracy and runs at ~15 ms/image vs dlib CNN's ~200 ms on CPU.

### Results
- 12/12 images: correct face count
- 0 missed faces, 0 false positives
- Avg detection time: ~15 ms (MediaPipe), ~53 ms (Haar fallback)

### Cross-questions
**Q: Why use two BlazeFace models?**  
They have different strengths at different face sizes and distances. Short-range handles close-up faces well; full-range handles smaller/more distant faces. Running both and taking the best detections improves recall.

**Q: Why run both original and flipped image?**  
BlazeFace is slightly asymmetric in its response to left vs right profiles. Flipping effectively doubles the detector's view.

**Q: Can the skin filter reject valid dark-skinned faces?**  
The majority vote (2 of 3 rules) is more robust than intersection for darker skin — at least the HSV and normalised-RGB rules typically still fire. The skin filter is a *scoring assist*, not a hard gate for MediaPipe detections (a high-confidence MediaPipe detection is kept even with low skin coverage).

**Q: What if MediaPipe model files are missing?**  
The code gracefully falls back to Haar cascades via a `try/except` at import time. The entire Haar fallback is fully implemented and achieves 12/12 on the sample images.

---

## Task 3 — Facial Landmark Detection

### Three landmarks required
- **Right eye centre** (subject's right = left side of image)
- **Left eye centre** (subject's left = right side of image)
- **Nose tip**

### Primary: MediaPipe FaceLandmarker

Model file: `face_landmarker.task` (3.6 MB). Produces 478 3-D landmarks.

- **Eyes:** landmarks 468 and 473 (iris centres). Sorted by x-coordinate: smaller x = subject's right eye, larger x = subject's left eye.
- **Nose tip:** landmark 1 (nose bridge tip in the canonical mesh).

**Why run on a padded ROI, not the full image?**  
The spec requires landmarks detected *within* the face box. Running FaceLandmarker on the full image would find the global face, not the specific detected face box. I crop a padded ROI (20% of max face dimension on each side) so the model sees only that face.

**Upscaling small ROIs:**  
ROIs smaller than 256 px on the short side are upscaled (up to 3×) before passing to the model. This improves landmark precision on small faces.

**Landmark validation (all must pass or fall back to Hough):**
1. Eye separation: 16–78% of face width
2. Average eye y: within −15% to +70% of face box height
3. Each eye individually: sits in the top 8–60% of the face box (catches forehead misplacements)
4. Vertical gap between the two eyes: < 20% of face height (catches cases where one eye slipped to the forehead)
5. Nose must be below the eyes and within 90% of face height

### Fallback: Hough circle transform (Lab 04)

When MediaPipe fails validation, I use the **Hough circle transform** on the upper 58% of the face ROI:

```python
cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                 dp=1.2, minDist=face_w//3,
                 param1=50, param2=12,
                 minRadius=int(face_w*0.06),
                 maxRadius=int(face_w*0.20))
```

**Scoring each circle (Lab 04 edge-overlap technique):**
1. Draw the circle on a blank mask
2. Count how many circle pixels coincide with Canny edges → `edge_overlap_score`
3. Add a `vertical_position_score` rewarding circles near 40% depth in the upper region

The highest-scoring circle in the left half = subject's right eye. Highest in right half = left eye.

**Why the upper 58% only?**  
Eyes are always in the upper half of a face. Including the lower half introduces circles from nostrils, mouth, chin that score well on edge overlap.

**Further fallback cascade:**
1. `haarcascade_eye_tree_eyeglasses.xml` — works with glasses
2. `haarcascade_righteye_2splits.xml` on the left half of the face
3. `haarcascade_lefteye_2splits.xml` on the right half
4. Geometric proportional estimate: (x + fw×0.33, y + fh×0.38) for right eye

**Nose tip:**  
Always found as the **intensity minimum** of a Gaussian-blurred patch in the central 30–70% (horizontal) × 50–72% (vertical) region. The shadow cast beneath the nose tip creates a reliable local dark spot.

**Eye separation sanity check:**  
If `|left_x − right_x| / face_width` is outside [0.18, 0.65], both eye estimates are replaced with geometric estimates. This prevents a badly rotated alignment from a plausible-looking but wrong landmark pair.

### Why not dlib 68-point shape predictor?

The 68-point predictor requires `shape_predictor_68_face_landmarks.dat` — a **68 MB** model file. This is too large to include in the submission zip. MediaPipe's `face_landmarker.task` (3.6 MB) gives sub-pixel iris precision and is small enough to submit.

### Results
- 26/26 faces: 100% both-eyes detected
- 100% all three landmarks inside the face bounding box
- 0 suspicious eye separations

### Cross-questions
**Q: Why use iris landmarks (468/473) instead of corner landmarks?**  
Iris centres are more stable across gaze directions. Corner landmarks (inner/outer canthus) shift with gaze. For alignment, the eye centre is the geometrically meaningful point.

**Q: How do you know which eye is right and which is left?**  
After detecting both iris centres, I sort by x-coordinate. Smaller x = left side of image = subject's right eye (anatomical perspective). This is consistent with the target: right eye at x=40 (left of canvas), left eye at x=85 (right of canvas).

**Q: What if HoughCircles finds no circles?**  
Falls through the cascade chain: glasses-aware cascade → directional cascades → geometric estimate. The geometric estimate always returns a valid point so alignment never fails.

**Q: Why Gaussian blur before Hough?**  
Hough circle detection is sensitive to noise. Gaussian blur (`σ=1.5`) smooths pixel noise while preserving the circular iris boundary gradient.

---

## Task 4 — Face Alignment

### What it must do
Apply **align → crop → resize** in the correct order so the output is a 125×125 image with:
- Subject's right eye at pixel (40, 40)
- Subject's left eye at pixel (85, 40)
- Nose tip near (63, 70)

### Why this order matters
If you crop first, rotating the crop introduces black borders. If you resize first, you lose resolution before aligning. `warpAffine` with output size (125,125) does all three in one operation — the transform simultaneously rotates, scales, translates, and produces the correctly sized output.

### My approach: closed-form 2-point similarity transform

A **similarity transform** has 4 degrees of freedom: scale, rotation angle, x-translation, y-translation.

I compute it **analytically from just the two eye points**:

```
src_vec = left_eye − right_eye        (eye-to-eye vector in source)
dst_vec = TARGET_LEFT − TARGET_RIGHT  = (85,40) − (40,40) = (45, 0)

scale = |dst_vec| / |src_vec|         (how much to scale up/down)
angle = arctan2(dst_vec) − arctan2(src_vec)   (how much to rotate)

Build the 2×3 matrix M from scale and angle,
translate so that right_eye maps exactly to (40,40).
```

### Why 2 points, not 3?

**I identified and fixed a tilt bug in the initial approach.** The original code used `cv2.estimateAffinePartial2D(src, dst, method=LMEDS)` with all 3 landmarks (right eye, left eye, nose).

`estimateAffinePartial2D` with 3 pairs **over-determines** the 4-DOF model (6 equations, 4 unknowns). When the nose mapping is geometrically inconsistent with the eye mapping — which happens on smiling or slightly turned faces — the LMEDS solver introduces a **spurious rotation** to minimise the combined error, visibly tilting the aligned crop even when the eyes are at the same height.

**Example:** Monica in the sample images had eyes at y=133 and y=133 (same height, no rotation needed), but the nose at x=130 maps to target x=63 — inconsistent with the left eye at x=132 mapping to x=85. The solver rotated the face ~20° to split the error.

**Fix:** 2 points give an exact, closed-form solution. Eyes land precisely at their target positions. The nose naturally falls near (63,70) since the face geometry is preserved by the similarity transform.

### `cv2.BORDER_REPLICATE`
Fills out-of-bounds pixels with the nearest valid image pixel. This prevents black borders when a face is near the image edge.

### Cross-questions
**Q: Is using `cv2.warpAffine` "writing code"?**  
Yes. I derive the transform matrix myself from first principles (scale, angle, translation computed analytically). `warpAffine` is a pixel interpolation utility, not a face alignment library. It's equivalent to using `numpy.dot` for matrix multiplication — I wrote the math, the library just applies it.

**Q: Why not use `cv2.getRotationMatrix2D`?**  
`getRotationMatrix2D` rotates around a centre point but doesn't handle non-uniform scaling or arbitrary source-to-target point mapping. The closed-form approach I implemented handles all cases including when the inter-eye distance needs to be scaled up or down.

**Q: The spec says nose at (63,70) — do you guarantee that?**  
The similarity transform is fitted to the eyes only, so the nose is not *constrained* to (63,70). However, because the transform preserves face geometry (it's a rigid motion + uniform scale), a correctly detected nose will land close to (63,70) naturally. The spec's target positions were derived from the same face geometry assumption.

**Q: What if `align_face` returns None?**  
The face is skipped for the corner thumbnail and embedding, but the bounding box and landmarks are still drawn on the output image.

---

## Task 5 — Identity Clustering

### Pipeline
```
For each image in folder:
  1. detect_faces → detect_landmarks → align_face  (Tasks 2–4)
  2. get_face_embedding → 128-D L2-normalised vector
Then:
  3. DBSCAN clustering on all embeddings
  4. Post-process: merge close singletons
  5. Save Identity_N_face_M.jpg
```

### Feature extraction: dlib ResNet embeddings

Via the `face_recognition` library (lazy import — only loaded when Task 5 runs).

- **Model:** dlib's ResNet-34 trained on 3M face images
- **Output:** 128-D float32 vector, L2-normalised
- **Accuracy:** 99.38% on LFW benchmark
- **Same-person distance:** typically < 0.6 in Euclidean space

The 125×125 aligned crop is passed with `locs=[(0,125,125,0)]` (full frame location) so face_recognition's internal 68-point landmark alignment runs on the correct region.

**Why face_recognition over raw dlib?**  
`face_recognition` wraps dlib's `face_encodings` with a clean API. The spec explicitly permits it: *"You are allowed to use a pre-trained model for this."*

### Why not use dlib directly?

`face_recognition` IS dlib — it's just a Python wrapper I chose for its clean API. There is no functional difference.

### Why not use LBP features instead?

I implemented a full LBP descriptor (5×5 spatial grid, 256-bin histogram per cell, 6400-D vector, L2-normalised, computed entirely in NumPy) and tested it first. Empirical result: cosine distances between **different people's** LBP descriptors were nearly identical (0.30–0.55), making no eps value reliably separate identities. The dlib embedding is far more discriminative because it was trained specifically to minimise intra-class variation.

LBP remains in the code as a **dependency-free fallback** for machines without face_recognition/dlib.

### Why not use other embeddings (FaceNet, ArcFace)?

These require TensorFlow or PyTorch plus large model files. The dlib ResNet via `face_recognition` has no deep learning framework dependency beyond the dlib library itself, which pip installs without GPU requirements.

### Clustering: DBSCAN

**Why DBSCAN over k-means?**  
k-means requires specifying k in advance. Since the number of unique identities is unknown, DBSCAN is the right choice — it finds clusters automatically. The spec also explicitly names DBSCAN as an option.

**Parameters:**
- `eps = 0.38` — maximum Euclidean distance for two embeddings to be in the same cluster
- `min_samples = 1` — every face forms at least one cluster; no face is discarded as noise

**How eps=0.38 was chosen:**  
I swept eps from 0.35 to 0.45 on the 12 sample images (26 faces) and computed the silhouette score at each value:

| eps | Clusters | Silhouette |
|-----|----------|------------|
| 0.35 | 9 | 0.191 |
| **0.38** | **6** | **0.214 ← selected** |
| 0.40 | 4 | 0.123 |
| 0.43 | 2 | 0.142 |
| 0.45 | 1 | 0.000 |

eps=0.38 gives the highest silhouette score (0.214). 6 clusters from 5 distinct people — one character is slightly over-split because their appearance changes significantly between images (different lighting/pose).

**Singleton merge post-processing:**  
After DBSCAN, isolated one-face clusters are merged into the nearest compatible cluster if:
- Euclidean distance < 0.515
- No two faces in the merged cluster come from the same source image (prevents merging different faces from one photo)

### Cross-questions
**Q: Why L2-normalise the embeddings?**  
L2-normalisation projects all embeddings onto the unit hypersphere. Euclidean distance between two L2-normalised vectors is equivalent to `sqrt(2 - 2*cos(angle))` — it becomes monotonically related to cosine similarity. This makes the eps threshold in DBSCAN interpretable and consistent regardless of the embedding magnitude.

**Q: How does DBSCAN decide cluster boundaries?**  
DBSCAN marks a point as a core point if it has at least `min_samples` neighbours within `eps` radius. Core points expand clusters by chaining. With `min_samples=1`, every point is a core point, so DBSCAN degenerates to single-linkage clustering — any two faces within eps of each other (or connected through a chain) end up in the same cluster.

**Q: What if face_recognition is not installed?**  
`get_face_embedding` catches the `ImportError` and returns None. `bulk_process_folder` then falls back to LBP features and DBSCAN with cosine distance (eps=0.30). The code never crashes due to missing dependencies.

**Q: Why 128 dimensions? Isn't that a lot?**  
128-D is actually compact for face embeddings. The dlib model was specifically trained to pack all identity-relevant information into 128 dimensions while discarding lighting, pose, and expression variation. Higher dimensions (e.g., ArcFace uses 512-D) can be more discriminative but require more data and compute to train.

---

## Why dlib is mentioned in the report

| Where | What is said |
|---|---|
| Task 3 alternatives | dlib 68-point shape predictor: 99%+ accuracy but requires a **68 MB model file** not included in the submission → rejected on practicality grounds |
| Task 5 feature extraction | dlib ResNet **is used** (via face_recognition library), explicitly permitted by the spec |

**dlib is not used for face detection (Task 2)** because MediaPipe BlazeFace is faster (~15 ms vs ~200 ms for dlib CNN) and more accurate on the sample images. dlib's HOG detector has lower recall on non-frontal faces.

---

## Overall Results Summary

| Metric | Result |
|---|---|
| Detection | 12/12 images, all face counts correct |
| Both eyes detected | 26/26 — 100% |
| All landmarks inside box | 100% |
| Alignment success | 26/26 — all exactly 125×125 px |
| Alignment tilt | Fixed — analytical 2-point transform |
| DBSCAN clusters | 6 (eps=0.38, silhouette=0.214) |
| Avg pipeline time | ~35 ms/image (MediaPipe) |
| Haar-only fallback | ~53 ms/image |

---

## Likely Demo Questions

**Q: Run it on this image.**  
The pipeline handles any 640×480 image with 1–4 faces. MediaPipe's models are trained on diverse datasets. The Haar fallback covers the case where model files aren't present.

**Q: What if there are 0 or 5 faces?**  
0 faces: returns an empty list; the output image is drawn without annotations. 5 faces: the cap-at-4 logic keeps the 4 highest-confidence detections. The spec guarantees 1–4 so this won't happen on test images.

**Q: How long does it take?**  
Single image: ~35 ms (0.035 seconds) including all 5 tasks. Bulk folder of 12 images: ~0.5 seconds for detection + landmarks + alignment, plus ~2–3 seconds for embedding computation (dlib runs on CPU).

**Q: What's the biggest weakness?**  
The skin filter can occasionally reject a valid face if lighting makes skin appear far from the expected hue range (e.g., very strong blue-tinted lighting). The fallback structure means the face is still usually detected via the Haar path.

**Q: Did you train any models?**  
No training. All models (BlazeFace, FaceLandmarker, dlib ResNet) are pretrained. The spec permits pretrained models for Tasks 2, 3, and 5. Task 4 (alignment) is coded from first principles.
