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
    """Render a compact iteration table: (Version, Change, Outcome)."""
    tbl = doc.add_table(rows=1, cols=3)
    tbl.style = 'Table Grid'
    hdr = tbl.rows[0].cells
    headers = ['Version', 'Change', 'Outcome']
    for i, htext in enumerate(headers):
        hdr[i].text = htext
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].runs[0].font.size = Pt(9)
    for ver, change, outcome in rows:
        row = tbl.add_row()
        for i, v in enumerate([ver, change, outcome]):
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

# -- 1. Introduction + Baseline vs Final
h1('1. Introduction')
body(
    'This report describes the face detection and identity clustering system implemented '
    'for the CITS4402 project. The system processes 640×480 colour images (1–4 faces), '
    'detects faces, localises three landmarks (right eye, left eye, nose tip), aligns each '
    'face to a 125×125 crop, and clusters faces across a folder into unique identities. '
    'The pipeline uses MediaPipe BlazeFace and FaceLandmarker when available, with a fully '
    'self-contained fallback built on OpenCV Haar cascades and the Hough circle transform '
    '(Lab 04). For clustering, 128-D dlib ResNet embeddings (via face_recognition) feed '
    'DBSCAN. The table below shows the headline measured outcomes from the naive starting '
    'point to the final implementation; subsequent sections detail each task\'s development.'
)

h2('Baseline vs Final (measured on the 12 sample images)')
tbl = doc.add_table(rows=1, cols=3)
tbl.style = 'Table Grid'
hdr = tbl.rows[0].cells
for i, htext in enumerate(['Metric', 'Naive baseline', 'Final implementation']):
    hdr[i].text = htext
    hdr[i].paragraphs[0].runs[0].bold = True
    hdr[i].paragraphs[0].runs[0].font.size = Pt(9)
for row in [
    ('Face detection',           '11/12 images (4.jpg missed)', '12/12 images, all counts correct'),
    ('Landmark detection',       '61% both-eyes (17/28 faces)', '100% both-eyes (26/26 faces)'),
    ('Landmarks inside box',     'not measured',                '100% RE, 100% LE, 100% nose'),
    ('Alignment success',        '61%, no similarity transform','100%, exact landmark targets (0 px residual)'),
    ('Identity clustering',      'not implemented',             'eps=0.38, 6 clusters from 26 faces'),
]:
    r = tbl.add_row()
    for i, v in enumerate(row):
        r.cells[i].text = v
        r.cells[i].paragraphs[0].runs[0].font.size = Pt(9)

# -- 2. Face Detection
h1('2. Face Detection (Task 2)')

h2('Development Iterations')
iteration_table([
    ('v1 (baseline)',
     'Haar cascade (haarcascade_frontalface_default) only, no skin filter',
     '11/12 images; 4.jpg missed entirely'),
    ('v2',
     'Added MediaPipe BlazeFace (short-range + full-range) as primary detector; '
     'Haar retained as fallback',
     '12/12 images, but false positives: 5 boxes on 11.jpg (small body region), '
     '1 face missed on 12.jpg (Ross failed cascade re-verification)'),
    ('v3',
     'Added skin-coverage bypass for verification (skin ≥ 75% AND box ≥ 60 px) — '
     'trust strong skin signal over cascade re-run; cap result at 4 (spec maximum)',
     'Fixed 12.jpg (Ross retained) and 11.jpg (small body box dropped); 12/12 correct'),
    ('v4',
     'Replaced YCrCb-only skin filter with three-rule majority vote '
     '(YCrCb + HSV wrap-around + normalised-RGB)',
     'More tolerant of unusual lighting/darker skin without losing FP rejection'),
    ('v5 (final)',
     'CLAHE replaces equalizeHist for the Haar-fallback grayscale input',
     '12/12 maintained; better local detail for the Haar fallback path'),
])

h2('Final Method')
body(
    'Primary: MediaPipe BlazeFace short-range (230 KB) + full-range (1.1 MB) TFLite models, '
    'run on the original and horizontally-flipped image. Results merged with IoU-NMS '
    '(threshold 0.28, containment > 72%). Each box expanded 20% up, 10% left/right, '
    '15% down to capture the full head. Capped at 4 by ranking on confidence + skin '
    'coverage score. Fallback when MediaPipe unavailable: haarcascade_frontalface_default '
    '+ alt2, 28% upward expansion, ROI re-verification, profile cascade as last resort.'
)

h2('Skin-Colour Segmentation (Mandatory)')
body(
    'Three-rule majority vote (Lab 05 technique): a pixel is skin when ≥ 2 of three rules '
    'agree — YCrCb (Cr ∈ [133,183], Cb ∈ [75,135]), HSV (H ∈ [0,28] ∪ [160,179]), '
    'normalised RGB (R > G−5, R > B, R > 70). Morphological open + close (5×5 ellipse) '
    'before the 20% pixel-ratio threshold.'
)

h2('Alternatives Considered')
bullet('dlib HOG face detector: lower recall on non-frontal faces vs BlazeFace.')
bullet('OpenCV YuNet: needs a separate ONNX file not bundled with opencv-python.')

# -- 3. Landmark Detection
h1('3. Facial Landmark Detection (Task 3)')

h2('Development Iterations')
iteration_table([
    ('v1 (baseline)',
     'haarcascade_eye.xml only',
     '61% both-eyes (17/28); fails on tilted, glasses, occluded faces'),
    ('v2',
     'Hough circle transform with Lab 04 edge-overlap scoring on upper 58% ROI; '
     'haarcascade_eye_tree_eyeglasses + directional eye cascades as chained fallbacks; '
     'nose tip via intensity minimum of Gaussian-blurred patch (not fixed proportion)',
     '100% both-eyes on all 26 faces'),
    ('v3',
     'Added MediaPipe FaceLandmarker (iris landmarks 468/473, nose landmark 1) '
     'as primary path',
     '100% maintained; sub-pixel iris precision'),
    ('v4',
     'Per-eye position validation: reject MediaPipe landmarks where any single eye is '
     'outside the top 8–60% of the face box, or eyes differ in y by > 20% face height',
     'Caught a Chandler-in-7.jpg case where MediaPipe placed the right eye on the '
     'forehead, causing a badly tilted alignment'),
    ('v5 (final)',
     'Integral-projection profile + symmetry-mirror fallback tiers; CLAHE preprocessing',
     '100% even with MediaPipe disabled (verified by forcing the landmarker off)'),
])

h2('Final Method')
body(
    'Seven-tier fallback chain (each tier only runs if the previous failed): '
    '(1) MediaPipe FaceLandmarker on a padded ROI; (2) Hough circles with Lab 04 '
    'edge-overlap scoring; (3) haarcascade_eye_tree_eyeglasses; (4) directional eye '
    'cascades; (5) integral-projection profile (darkest row in upper face); '
    '(6) symmetry mirror if one eye is found; (7) proportional geometric estimate. '
    'Nose tip always from intensity minimum in the central 30–70% × 50–72% region.'
)

h2('Alternatives Considered')
bullet('dlib 68-point shape predictor: 68 MB model file, too large for submission.')

# -- 4. Alignment
h1('4. Facial Alignment (Task 4)')

h2('Development Iterations')
iteration_table([
    ('v1 (baseline)',
     'getRotationMatrix2D for rotation, then separate crop + resize. '
     'Does not use a similarity transform; landmarks not at target positions',
     '61% success (tied to landmark failures)'),
    ('v2',
     'cv2.estimateAffinePartial2D with LMEDS on all three landmarks',
     '100% success but visible tilt artefact when nose constraint is geometrically '
     'inconsistent with eyes (e.g., Monica in 7.jpg, eyes both at y=133 but face '
     'aligned with ~20° rotation)'),
    ('v3',
     'Analytical 2-point similarity transform from the two eye points only',
     '100% no tilt; eyes exact at (40,40) and (85,40); but nose drifted from (63,70) '
     '— flagged by the lab facilitator'),
    ('v4',
     'Least-squares similarity transform over all three landmarks (closed form via '
     'np.linalg.lstsq on the 6-equation 4-unknown system)',
     '100%; reduced tilt vs LMEDS but still measured max 9.5 px residual on the '
     'sample set because a 4-DOF model cannot exactly fit 3 arbitrary point pairs'),
    ('v5 (final)',
     'cv2.getAffineTransform with all three landmarks',
     '100%; verified 0 px residual on all 26 faces — every landmark lands precisely '
     'at its target coordinate'),
])

h2('Final Method')
body(
    'cv2.getAffineTransform takes exactly 3 source→target pairs and returns the unique '
    '6-DOF affine matrix that maps them exactly. A single cv2.warpAffine call with output '
    'size (125, 125) simultaneously aligns, crops, resizes, and (where needed) lightly '
    'stretches the face so right_eye → (40, 40), left_eye → (85, 40), '
    'nose_tip → (63, 70) are all exact. borderMode=BORDER_REPLICATE fills out-of-bounds '
    'pixels, preventing black borders. Landmark dots on the corner thumbnail are drawn '
    'at the fixed target coordinates and never move between images.'
)

# -- 5. Identity Clustering
h1('5. Identity Clustering (Task 5)')

h2('Development Iterations')
iteration_table([
    ('v1 (baseline)',
     'Not implemented in the baseline code',
     '—'),
    ('v2',
     '128-D dlib ResNet embeddings via face_recognition (explicitly permitted by spec); '
     'DBSCAN with min_samples=1 and Euclidean distance',
     'Clusters identities meaningfully; see eps sweep below'),
    ('v3',
     'Singleton-merge post-processing: merge isolated one-face clusters into the nearest '
     'compatible cluster if distance < 0.515 AND no shared source image',
     'Repairs cases where one image of a person has unusual pose/lighting'),
    ('v4',
     'Upgraded the LBP fallback descriptor (used when face_recognition is unavailable) '
     'from basic 256-bin 5×5 to LBP-u2 7×7 multi-scale (Ahonen et al. 2006): '
     '59 uniform-pattern bins × 49 cells × 2 radii = 5782-D L2-normalised',
     'LBP fallback silhouette at peak: 0.04 → 0.11 (≈ 3× improvement)'),
])

h2('DBSCAN eps Sweep (final code, 26 faces)')
tbl = doc.add_table(rows=1, cols=3)
tbl.style = 'Table Grid'
hdr = tbl.rows[0].cells
hdr[0].text = 'eps'; hdr[1].text = 'Clusters'; hdr[2].text = 'Silhouette'
for r in hdr:
    r.paragraphs[0].runs[0].bold = True
    r.paragraphs[0].runs[0].font.size = Pt(9)
for eps, nc, sil in [
    (0.35, 10, 0.1408),
    (0.38, 6,  0.1021),
    (0.40, 4,  0.0954),
    (0.43, 1,  0.0000),
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

body(
    'eps=0.38 was selected because 6 clusters is closest to the 5-identity ground truth '
    '(the 5 sample-image characters), with one slight over-split for a character whose '
    'appearance varies significantly between images. eps=0.35 has marginally higher '
    'silhouette (0.14) but over-splits to 10 clusters. eps=0.40 collapses two distinct '
    'identities. min_samples=1 ensures every face joins a cluster (no noise points).'
)

h2('Alternatives Considered')
bullet('k-means: cannot be used — number of identities is unknown a priori.')
bullet('Agglomerative clustering: rejected because DBSCAN is explicitly named in the spec.')

# -- 6. Final Results
h1('6. Final Results')
body('Measured on the 12 sample images (26 detected faces, all face counts match ground truth):')
bullet('Face detection: 12/12 images correct; avg detection ≈ 39 ms (MediaPipe path).')
bullet('Landmark detection: 100% both eyes; 100% RE, 100% LE, 100% nose inside the face box.')
bullet('Alignment: 100% success; 0 px residual on every landmark; all outputs exactly 125×125.')
bullet('Clustering: 6 DBSCAN clusters from 26 faces (eps=0.38, silhouette = 0.10).')
body(
    'Known limitations: (1) MediaPipe model files (≈ 4.9 MB total) must accompany the '
    'script — the Haar/Hough fallback covers their absence automatically. (2) face_recognition '
    '(dlib) must be pip-installed for best clustering — the LBP-u2 fallback works without '
    'it but with reduced discrimination. (3) eps=0.38 over-splits one identity (6 clusters '
    'from 5 characters); eps=0.40 merges two distinct characters, so 6 is the better error.'
)

doc.save('/Users/preet/Documents/GitHub/face-detection/Report.docx')
print('Report.docx saved.')
