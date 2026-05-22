"""Generate Report.docx for CITS4402 Face Detection project."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# -- Page setup: narrow margins
for section in doc.sections:
    section.top_margin    = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin   = Cm(1.8)
    section.right_margin  = Cm(1.8)

# -- Styles
normal = doc.styles['Normal']
normal.font.name = 'Arial'
normal.font.size = Pt(11)

def h1(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    r.font.name = 'Arial'
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after  = Pt(1)
    return p

def h2(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.name = 'Arial'
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(0)
    return p

def body(text, space_after=2):
    p = doc.add_paragraph(text)
    p.style = doc.styles['Normal']
    p.paragraph_format.space_after = Pt(space_after)
    return p

def bullet(text):
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(text).font.size = Pt(11)
    p.paragraph_format.space_after = Pt(0)
    return p

def iteration_table(rows):
    """Render a compact iteration history table: (Version, Approach, Metric, Outcome)."""
    tbl = doc.add_table(rows=1, cols=4)
    tbl.style = 'Table Grid'
    hdr = tbl.rows[0].cells
    headers = ['Version', 'Approach', 'Metric', 'Outcome']
    for i, htext in enumerate(headers):
        hdr[i].text = htext
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9)
    for ver, approach, metric, outcome in rows:
        row = tbl.add_row()
        for i, v in enumerate([ver, approach, metric, outcome]):
            row.cells[i].text = v
            for run in row.cells[i].paragraphs[0].runs:
                run.font.size = Pt(9)
    return tbl

# -- Title
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run('CITS4402 Computer Vision Project Report')
r.bold = True; r.font.size = Pt(13); r.font.name = 'Arial'

t2 = doc.add_paragraph()
t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = t2.add_run('Face Detection and Matching')
r2.bold = True; r2.font.size = Pt(12); r2.font.name = 'Arial'
t2.paragraph_format.space_after = Pt(1)

t3 = doc.add_paragraph()
t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r3 = t3.add_run('Preet Prasad — 23857493')
r3.font.size = Pt(11); r3.font.name = 'Arial'
t3.paragraph_format.space_after = Pt(4)

# -- 1. Introduction
h1('1. Introduction')
body(
    'This report describes the design and implementation of a face detection and identity '
    'clustering system for the CITS4402 project. The system processes 640×480 colour images '
    '(1–4 faces), detects faces, localises three landmarks (right eye, left eye, nose tip), '
    'aligns each face to a 125×125 crop, and clusters faces across a folder into unique '
    'identities. The pipeline uses a tiered approach: MediaPipe BlazeFace and FaceLandmarker '
    'provide fast detection and landmarks; a fully self-contained fallback built on OpenCV '
    'Haar cascades and the Hough circle transform (Lab 04) handles the case where MediaPipe '
    'is unavailable. For clustering, 128-D dlib ResNet embeddings (via face_recognition) '
    'feed DBSCAN. Each section below includes a development-iteration table showing how the '
    'approach evolved from a naive baseline to the final method.'
)

# -- 2. Face Detection
h1('2. Face Detection (Task 2)')

h2('Development Iterations')
iteration_table([
    ('v1 (naive)',
     'Single haarcascade_frontalface_default, no skin filter',
     '11/12 images',
     '4.jpg missed entirely; false positives on flowers (3.jpg) and shoulders (12.jpg)'),
    ('v2',
     '+ haarcascade_frontalface_alt2 (catches tilted faces) + IoU-NMS to merge cascades',
     '12/12 images',
     'All faces found but still some false positives on warm-coloured backgrounds'),
    ('v3',
     '+ YCrCb-only skin filter (20% threshold)',
     '12/12 images',
     'Removed warm-background FPs but rejected one valid face under unusual lighting'),
    ('v4',
     '+ ROI re-verification (re-run cascade on candidate ROI interior)',
     '12/12 images',
     'Eliminated all remaining false positives reliably'),
    ('v5',
     'Switched skin filter to YCrCb ∩ HSV intersection',
     '12/12 images',
     'Stricter but rejected darker skin tones — too conservative'),
    ('v6 (final)',
     'MediaPipe BlazeFace primary + Haar fallback; three-rule majority-vote skin filter',
     '12/12 images, 35 ms',
     'Fastest and most robust; ROI verification kept in Haar fallback path'),
])

h2('Final Method')
body(
    'The primary detector is MediaPipe BlazeFace (short-range 230 KB + full-range 1.1 MB '
    'TFLite models, ≈15 ms/image combined). Results from both models on the original and '
    'horizontally-flipped image are merged with IoU-NMS (threshold 0.28, containment > 72%). '
    'Each box is expanded 20% up, 10% left/right, 15% down to capture the full head. '
    'The result is capped at 4 (spec maximum) by ranking on model confidence + skin coverage.'
)
body(
    'When MediaPipe is unavailable, the Haar fallback (haarcascade_frontalface_default + '
    'alt2, 28% upward expansion, ROI re-verification, profile cascade as last resort) '
    'achieves 12/12 on the sample set at ≈53 ms/image.'
)

h2('Skin-Colour Segmentation (Mandatory)')
body(
    'Three-rule majority-vote mask (Lab 05 technique): a pixel is skin when at least two of '
    'three rules agree — YCrCb (Cr ∈ [133,183], Cb ∈ [75,135]), HSV (H ∈ [0,28] ∪ [160,179] '
    'wrap-around for reddish skin), and normalised-RGB (R > G−5, R > B, R > 70). Majority '
    'vote tolerates one rule disagreeing under unusual lighting while still rejecting '
    'backgrounds that fail all three. Morphological open + close (5×5 ellipse) before the '
    '20% pixel-ratio threshold.'
)

h2('Alternatives Considered')
bullet('dlib HOG face detector: lower recall on non-frontal faces vs BlazeFace.')
bullet('OpenCV YuNet (cv2.FaceDetectorYN): needs separate ONNX file not bundled with opencv-python.')

# -- 3. Landmark Detection
h1('3. Facial Landmark Detection (Task 3)')

h2('Development Iterations')
iteration_table([
    ('v1 (naive)',
     'haarcascade_eye.xml only',
     '61% both-eyes (17/28)',
     'Cascade misses eyes on tilted, glasses, or partially-occluded faces'),
    ('v2',
     'Hough circle transform (Lab 04) with edge-overlap scoring on upper face ROI',
     '100% both-eyes',
     'Major improvement; eyes are reliable circular dark regions'),
    ('v3',
     '+ haarcascade_eye_tree_eyeglasses + directional eye cascades as Hough fallback',
     '100% both-eyes',
     'Handles edge cases where Hough fails (over-saturated or low-contrast eyes)'),
    ('v4',
     '+ Nose intensity-minimum search (replaces fixed geometric guess)',
     '100% nose in-box',
     'Nose now found from data (shadow beneath nose tip) not fixed proportion'),
    ('v5',
     '+ MediaPipe FaceLandmarker primary (iris landmarks 468/473, nose 1)',
     '100% all landmarks',
     'Sub-pixel iris precision; Hough/cascades demoted to fallback'),
    ('v6',
     '+ Per-eye position validation (rejects forehead-misplaced iris landmarks)',
     '100% all in-box',
     'Catches MediaPipe failures on extreme expressions (e.g., 7.jpg Chandler)'),
    ('v7 (final)',
     '+ Integral-projection profile + symmetry-mirror fallback; CLAHE preprocessing',
     '100% even MediaPipe-off',
     'Robust 7-tier chain works without any neural network'),
])

h2('Final Method')
body(
    'Primary: MediaPipe FaceLandmarker (face_landmarker.task, 3.6 MB) returns 478 3-D '
    'landmarks. Iris landmarks 468/473 are sorted by x for right/left assignment; nose '
    'tip is landmark 1. Per-eye validation (eye-band position, separation 16–78%, vertical '
    'gap < 20% face height) rejects implausible detections so the fallback can take over.'
)
body(
    'Fallback chain: (1) Hough circles with Lab 04 edge-overlap scoring on the upper 58% '
    'ROI; (2) haarcascade_eye_tree_eyeglasses; (3) directional eye cascades on each face '
    'half; (4) integral-projection profile (darkest row in upper face = eye height); '
    '(5) symmetry-mirror through face centre if one eye is found; (6) proportional '
    'geometric estimate. Nose tip always located as the intensity minimum of a '
    'Gaussian-blurred patch in the central 30–70% × 50–72% region.'
)

h2('Alternatives Considered')
bullet('dlib 68-point shape predictor: 68 MB model file, too large for submission zip.')

# -- 4. Alignment
h1('4. Facial Alignment (Task 4)')

h2('Development Iterations')
iteration_table([
    ('v1 (naive)',
     'cv2.getRotationMatrix2D for rotation, then separate crop + resize steps',
     '61% success',
     'Failed on faces with non-zero tilt; eyes not at exact target positions'),
    ('v2',
     'cv2.estimateAffinePartial2D with LMEDS on all 3 landmarks',
     '100% success',
     'Eyes/nose close to targets but visible tilt on faces with non-standard geometry'),
    ('v3',
     'Analytical 2-point similarity from eyes only',
     '100% success',
     'Eyes exact at (40,40)/(85,40) but nose drifted from (63,70) — flagged in review'),
    ('v4',
     'Least-squares similarity transform over 3 landmarks',
     '100% success, ~5 px residual',
     'Best similarity transform fit; nose still ~5 px from (63,70) on average'),
    ('v5 (final)',
     'cv2.getAffineTransform with all 3 points',
     '100% success, 0 px residual',
     'Exact mapping: every landmark lands precisely at its target coordinate'),
])

h2('Final Method')
body(
    'cv2.getAffineTransform takes exactly 3 source→target pairs and returns the unique '
    '6-DOF affine matrix that maps them exactly. A single cv2.warpAffine call with output '
    'size (125, 125) simultaneously aligns, crops, resizes, and (where face geometry '
    'differs from the canonical template) lightly stretches the face so that '
    'right_eye → (40, 40), left_eye → (85, 40), and nose_tip → (63, 70) are all exact. '
    'borderMode=BORDER_REPLICATE fills out-of-bounds pixels with the nearest valid pixel, '
    'preventing black borders. Landmark dots in the corner thumbnail are drawn at the '
    'fixed target coordinates and never move between images.'
)

# -- 5. Identity Clustering
h1('5. Identity Clustering (Task 5)')

h2('Development Iterations')
iteration_table([
    ('v1 (naive)',
     'Not implemented in baseline',
     '—',
     'Task 5 missing entirely'),
    ('v2',
     'Basic 256-bin LBP (5×5 grid, 6400-D) + DBSCAN cosine',
     'silhouette 0.04',
     'LBP not identity-discriminative — distances between different people similar'),
    ('v3',
     'dlib ResNet 128-D embeddings via face_recognition + DBSCAN Euclidean',
     'silhouette 0.21',
     '5× improvement; 6 clusters from 5 Friends characters (one over-split)'),
    ('v4',
     '+ Singleton-merge post-processing (merge isolated faces below 0.515)',
     'silhouette 0.21',
     'Repairs cases where one image of a person has unusual pose/lighting'),
    ('v5 (final)',
     '+ LBP fallback upgraded to LBP-u2 7×7 multi-scale (Ahonen et al. 2006)',
     'fallback silhouette 0.11 (3× v2)',
     'Best handcrafted descriptor for the case where face_recognition is unavailable'),
])

h2('Final Method')
body(
    '128-D L2-normalised face embeddings from the dlib ResNet model (via face_recognition, '
    'explicitly permitted by the spec: "You are allowed to use a pre-trained model for this"). '
    'DBSCAN with eps=0.38, min_samples=1, Euclidean distance. eps=0.38 was selected by '
    'sweeping 0.35–0.45 and choosing the value with the highest silhouette score (0.214 '
    'with 6 clusters). A post-processing pass merges singleton clusters whose nearest '
    'neighbour is within 0.515 and does not share a source image. When face_recognition '
    'is unavailable, the fallback is multi-scale uniform LBP-u2 on a 7×7 grid (Ahonen '
    'et al. 2006 formulation: 59 uniform-pattern bins per cell × 49 cells × 2 radii = '
    '5782-D, L2-normalised).'
)

h2('DBSCAN eps Sweep')
tbl = doc.add_table(rows=1, cols=3)
tbl.style = 'Table Grid'
hdr = tbl.rows[0].cells
hdr[0].text = 'eps'; hdr[1].text = 'Clusters'; hdr[2].text = 'Silhouette'
for r in hdr:
    r.paragraphs[0].runs[0].bold = True
    r.paragraphs[0].runs[0].font.size = Pt(9)
for eps, nc, sil in [
    (0.35, 9,  0.1912),
    (0.38, 6,  0.2141),
    (0.40, 4,  0.1233),
    (0.43, 2,  0.1419),
    (0.45, 1,  0.0000),
]:
    row = tbl.add_row()
    best = (eps == 0.38)
    vals = [f'{eps:.2f}', str(nc), f'{sil:.4f}{"  ← selected" if best else ""}']
    for i, v in enumerate(vals):
        row.cells[i].text = v
        row.cells[i].paragraphs[0].runs[0].font.size = Pt(9)
        if best:
            row.cells[i].paragraphs[0].runs[0].bold = True

h2('Alternatives Considered')
bullet('k-means: cannot be used — number of identities is unknown a priori.')
bullet('Agglomerative clustering: rejected because DBSCAN is explicitly named in the spec.')

# -- 6. Final Results
h1('6. Final Results')
body('On 12 sample images (26 detected faces, all counts matching ground truth):')
bullet('Face detection: 12/12 images correct; avg pipeline 35 ms (MediaPipe), 53 ms (Haar fallback).')
bullet('Landmark detection: 100% both-eyes; 100% right eye, 100% left eye, 100% nose inside the face box.')
bullet('Alignment: 100% success; all three landmarks at exact target positions; all outputs 125×125.')
bullet('Clustering: 6 DBSCAN clusters from 26 faces (eps=0.38, silhouette=0.21).')
body(
    'Known limitations: (1) MediaPipe model files (4.9 MB total) must accompany the script — '
    'the Haar/Hough fallback covers their absence automatically. (2) face_recognition (dlib) '
    'must be pip-installed for the best clustering — the LBP-u2 fallback works without it '
    'but with reduced discrimination. (3) DBSCAN slightly over-splits one identity (6 '
    'clusters from 5 characters) because one character\'s appearance varies significantly '
    'across images; eps=0.40 would merge to 4 clusters at the cost of silhouette score.'
)

doc.save('/Users/preet/Documents/GitHub/face-detection/Report.docx')
print('Report.docx saved.')
