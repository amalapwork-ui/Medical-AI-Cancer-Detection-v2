# Medical AI Cancer Detection System

AI-powered multi-organ cancer detection from medical images. Classifies brain MRI, lung CT, and colon histopathology images using EfficientNetV2S deep learning models with a statistical confidence-based quality gate.

> **Disclaimer:** Research and educational use only. Not a substitute for professional medical diagnosis.

---

## Architecture

```
User (Browser)
    │
    │  select cancer type + upload image
    ▼
Streamlit Frontend  (port 8501)
    │
    │  HTTP POST /predict  (multipart form-data)
    ▼
FastAPI Backend  (port 8000)
    │
    ├─▶ 1. validate_input_bytes()        ← reject corrupted / oversized / wrong format
    │
    ├─▶ 2. preprocess_image()            ← RGB resize 224×224, CLAHE for lung CT
    │
    ├─▶ 3. screen_image_domain()         ← four-signal domain check (no extra model)
    │         Signal 1  — saturation ceiling (per-modality):
    │           brain/lung: saturation > 0.15 → "non_medical_image"
    │           colon:      saturation > 0.85 → "non_medical_image"
    │           (real H&E reaches 0.55–0.70; 0.85 is above any clinical range)
    │         Signal 2  — saturation floor (colon only):
    │           colon: saturation < 0.05 → "cross_modality" (CT/MRI to histopathology model)
    │         Signal 2b — flat-colour check (colon only, RGB-derived):
    │           mean_sat > 0.35 AND sat_std < 0.04 AND pixel_std < 0.04
    │                  → "non_medical_image" (solid-colour graphic, not tissue)
    │         Signal 3  — intensity profile (brain and lung models only):
    │           brain: dark_fraction > 0.50 AND pixel_std > 0.32 → "cross_modality" (lung CT)
    │           lung:  centre_std < 0.08 AND centre_mean > 0.40 AND dark_fraction > 0.15
    │                  → "cross_modality" (brain MRI)
    │         (edge density evaluated + removed: causes false rejections on HRCT/nodule CT)
    │
    ├─▶ 4. classify_cancer()             ← route to brain / lung / colon model
    │         │
    │         └── _assess_prediction_quality()  — four-criterion quality gate:
    │               conf < 0.35 or > 99%  → predicted_class: "Invalid Scan"
    │               margin < 0.07         → prediction_status: "ambiguous"
    │               norm_H/H_max > 0.95   →  (distribution too flat)
    │               norm_H/H_max < 0.01   →  (distribution unnaturally peaked = OOD)
    │
    └─▶ 5. JSON response                 ← predicted_class, confidence, all_probabilities
```

---

## Models

| Cancer | Architecture | Dataset | Classes | Accuracy |
|--------|-------------|---------|---------|----------|
| Brain  | EfficientNetV2S | masoudnickparvar/brain-tumor-mri-dataset | glioma, meningioma, notumor, pituitary | ~91% |
| Lung   | EfficientNetV2S | hamdallak/the-iqothnccd-lung-cancer-dataset (CT) | benign, malignant, normal | ~82–90% |
| Colon  | EfficientNetB2 | kmader/colorectal-histology-mnist (Kather 2016) | adipose, complex, debris, lympho, mucosa, stroma, tumor *(empty suppressed — returns "Invalid Scan")* | ~88–93% |

**Preprocessing note:** All models use `include_preprocessing=True`. Pass raw `[0, 255]` float32 — do **not** normalise to `[0, 1]`. The lung model additionally requires CLAHE (applied automatically by `predict.py`).

---

## Project Structure

```
backend/
  app/
    config.py     ← single source of truth: model paths, labels, thresholds
    schemas.py    ← Pydantic response models
    predict.py    ← 2-stage inference pipeline
    main.py       ← FastAPI app, lifespan model loading, POST /predict + GET /health
  models/         ← .keras files (tracked in repo)

frontend/
  streamlit_app.py ← Streamlit UI

training/
  train_brain.py      ← EfficientNetV2S, 2-phase, brain MRI dataset
  train_lung.py       ← EfficientNetV2S, focal loss, CLAHE, patient-level split
  train_colon.py      ← EfficientNetB2, group-aware patch split, MixUp
  utils/
    augmentation.py   ← shared augmentation strategies + MixUp
    evaluation.py     ← shared metrics, confusion matrix, calibration plots

tests/
  test_inference.py  ← unit tests: input validation, preprocessing, quality gate, pipeline
  test_api.py        ← FastAPI integration tests via TestClient
  test_models.py     ← model spec tests: output shapes, class counts, label routing
  demo.py            ← generates demo_test_images.zip
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### 1 — Clone and create virtual environment

```bash
git clone <repository-url>
cd medical-ai-cancer-detection
python -m venv myvenv

# Windows
myvenv\Scripts\activate

# macOS / Linux
source myvenv/bin/activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Place model files

`.keras` files are in `backend/models/` (tracked in repo):

```
backend/models/
├── brain_cancer_effnet.keras
├── lung_cancer_final.keras
└── colon_cancer_final.keras
```

Missing models are reported by `GET /health` — the server still starts.

---

## Running the System

Two terminals are required.

### Terminal 1 — Backend

```bash
# From the project root (not from backend/)
uvicorn backend.app.main:app --reload
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

> **Important:** Run from the **project root**, not from `backend/`. `config.py` resolves model paths relative to its own location; running from the wrong directory will cause `FileNotFoundError`.

### Terminal 2 — Frontend

```bash
streamlit run frontend/streamlit_app.py
```

- UI: `http://localhost:8501`

---

## API Usage

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@brain_scan.jpg;type=image/jpeg" \
  -F "cancer_type=brain"
```

**Successful response:**

```json
{
  "status": "success",
  "cancer_type": "brain",
  "predicted_class": "Glioma"
}
```

**Ambiguous response (low confidence or uncertain output):**

```json
{
  "status": "success",
  "cancer_type": "brain",
  "predicted_class": "Invalid Scan"
}
```

When the model cannot produce a reliable prediction (low confidence, near-uniform distribution, or anomalous output), `predicted_class` is set to `"Invalid Scan"`. The internal quality gate still runs on every request — its result is expressed through the class name, not exposed as a separate field.

---

## Out-of-Distribution (OOD) and Cross-Modality Handling

### Design principle: strict with invalid, tolerant with valid

The system is designed to accept valid medical images with high tolerance for real-world variation (contrast, brightness, noise, scan protocol) while firmly rejecting clearly invalid inputs. Strict rules that produce false rejections are worse than no rule at all — a radiologist's valid CT scan being turned away is an unacceptable failure mode.

### Why confidence alone fails

Softmax classifiers sum to 1.0 **regardless of input**. A logo fed to a brain model still produces a peaked distribution — one class always wins, even when none apply. Cross-modality inputs behave the same way: a lung CT submitted to the colon model will produce a colon tissue prediction, not an error. The model has no concept of "wrong modality."

### Four-signal domain screen (before model inference)

All signals exploit **physical imaging properties** rather than protocol-dependent measurements, making them robust to CT acquisition variation, window/level settings, and JPEG compression.

| Signal | Basis | Threshold | Catches | Rejection reason |
|--------|-------|-----------|---------|-----------------|
| 1 — Saturation ceiling | MRI/CT are grayscale by physics; H&E staining adds colour. Real H&E reaches 0.55–0.70 mean saturation. | brain/lung > 0.15; colon > 0.85 | Logos, photos, normalstained histopathology sent to brain/lung; extreme neon images sent to colon | `non_medical_image` |
| 2 — Saturation floor | H&E process always introduces pink/purple (even background tiles) | colon < 0.05 | Grayscale CT or MRI submitted to the colon model | `cross_modality` |
| 2b — Flat-colour check (colon only) | Real H&E slides have spatial staining variation and cell micro-texture; solid-colour graphics do not. Uses RGB-derived saturation proxy — no cv2 dependency. | mean_sat > 0.35 AND sat_std < 0.04 AND pixel_std < 0.04 (AND logic) | Logos, swatches, uniform colour fills with enough colour to pass Signal 1 | `non_medical_image` |
| 3 — Intensity profile | Lung CT has a bimodal histogram (dark parenchyma + bright bone); brain MRI has smooth uniform brain tissue in the centre with dark only at the periphery | brain: dark_fraction > 0.50 AND std > 0.32; lung: centre_std < 0.08 AND centre_mean > 0.40 AND dark_fraction > 0.15 | Lung CT submitted to brain model; brain MRI submitted to lung model | `cross_modality` |

**Why edge density (Laplacian variance) was evaluated and removed:** Edge density was intended to catch histopathology submitted to brain/lung models. In practice it caused false rejections of valid lung CT scans: high-resolution CT (HRCT), nodule-dense scans, and narrow-window JPEG exports routinely produce Laplacian variance above any reasonable fixed threshold. Unlike saturation — which is physically fixed by the imaging modality — edge density depends on CT protocol, window settings, and compression, making any single threshold brittle.

### Internal quality gate (after model inference, before response)

Probability values are computed on every request and used internally to evaluate prediction quality. They are not exposed in the API response — only the final `predicted_class` is returned.

Four criteria must all pass; any failure sets `predicted_class` to `"Invalid Scan"`:

| Criterion | Brain / Lung threshold | Colon threshold | What it catches |
|-----------|----------------------|-----------------|-----------------|
| Confidence floor | ≥ 0.35 | ≥ 0.20 | Near-random outputs |
| Margin | ≥ 0.07 | ≥ 0.02 | Near-ties between top-2 classes |
| Entropy ceiling | H/H_max ≤ 0.95 | H/H_max ≤ 0.99 | Flat, uncertain distributions |
| Entropy floor | H/H_max ≥ 0.01 | H/H_max ≥ 0.01 | True OOD over-commitment (p₁ ≳ 99.9%) |

**Why colon uses lenient thresholds:** The Kather 2016 8-class model covers low-information tissue classes (empty background, sparse stroma, adipose, cellular debris) that produce weaker softmax signals than tumour classes. Valid low-cellularity patches return conf 20–30% and norm_H 0.96–0.98 — above the 8-class random baseline (12.5%) but outside the global thresholds calibrated for 3- and 4-class models. The colon-specific thresholds accept these valid uncertain predictions while still rejecting truly random outputs (conf < 20%) and near-uniform distributions (norm_H > 0.99).

**Important — margin threshold calibration:** The global margin threshold is 0.07, not 0.10. The brain model's `notumor` class receives ~2× fewer training examples (~395 vs ~826 per tumour class) in the masoudnickparvar dataset; detecting the absence of a tumour is also inherently harder than detecting its presence. As a result, valid notumor predictions typically have margins of 7–12% — smaller than the 15–40% margins typical of positive-tumour classes. A threshold of 0.10 incorrectly rejected these valid predictions. 0.07 still filters true near-ties (< 7%) while accepting small-but-decisive notumor margins.

**Important — entropy floor:** The entropy floor is 0.01 for all modalities. For a 4-class model, the floor fires only when a single class absorbs more than ~99.9% of the probability mass — a regime that trained EfficientNetV2S models do not reach on valid in-distribution medical images. Setting the floor too high (e.g., 0.05) incorrectly rejects legitimate 99%+ confidence predictions on clear cases such as pituitary adenomas and normal brain scans.

### Cross-modality coverage

| Scenario | Primary gate | Result |
|----------|-------------|--------|
| Brain MRI → brain model | passes all screens | success |
| Lung CT → lung model | passes all screens | success |
| Colon H&E → colon model (any saturation 0.05–0.85) | passes all screens | success |
| Logo / photo → brain/lung model | Signal 1: saturation > 0.15 | rejected: `non_medical_image` |
| Logo / photo → colon model (neon, sat > 0.85) | Signal 1: saturation > 0.85 | rejected: `non_medical_image` |
| Solid-colour graphic → colon model (sat 0.35–0.85) | Signal 2b: flat texture detected | rejected: `non_medical_image` |
| H&E histopathology → brain/lung (normal stain) | Signal 1: saturation > 0.15 | rejected: `non_medical_image` |
| CT or MRI → colon model | Signal 2: saturation < 0.05 | rejected: `cross_modality` |
| Lung CT → brain model | Signal 3: high dark_fraction + high std | rejected: `cross_modality` |
| Brain MRI → lung model | Signal 3: smooth bright centre + dark periphery | rejected: `cross_modality` |
| H&E histopathology → brain/lung (light stain) | Quality gate: flat/anomalous distribution | success with `ambiguous` |

### Acknowledged limitations

1. **Greyscale non-medical images** (black-and-white logos, greyscale photos) pass Signal 1 and may pass Signal 3 if their intensity profile does not match a CT or MRI pattern. These rely on the softmax quality gate as the final fallback.
2. **Coloured natural photos submitted to colon** (skin close-ups, flowers) may have enough spatial variation to pass Signal 2b. The softmax quality gate handles these downstream — the colon EfficientNetB2 model produces a confused distribution on non-histopathology inputs.
3. **Unusual acquisition parameters** — extremely dark MRI sequences or reconstructed CT protocols that deviate significantly from standard distributions may occasionally produce intensity features that do not trigger Signal 3. The conservative AND-logic thresholds are calibrated to minimise false rejections of valid scans at the cost of accepting some edge-case cross-modality submissions.

---

## Training

All training scripts are designed for Google Colab (GPU required for reasonable speed). Each script downloads its own dataset via the Kaggle API and saves the model to `output/<model>/`. Copy the `.keras` file to `backend/models/` when done.

| Script | Dataset | Output |
|--------|---------|--------|
| `training/train_brain.py` | masoudnickparvar/brain-tumor-mri-dataset | `brain_cancer_effnet.keras` |
| `training/train_lung.py` | hamdallak/the-iqothnccd-lung-cancer-dataset | `lung_cancer_final.keras` |
| `training/train_colon.py` | kmader/colorectal-histology-mnist | `colon_cancer_final.keras` |

---

## Tests

```bash
# All tests (no GPU, no model files required)
pytest tests/ -v

# Unit tests only (inference logic)
pytest tests/test_inference.py -v

# API integration tests (FastAPI TestClient)
pytest tests/test_api.py -v

# Model spec tests (output shapes, label routing)
pytest tests/test_models.py -v

# Single test by name
pytest tests/test_inference.py::TestFullPipeline::test_valid_brain_scan -v
```

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: backend/models/…` | uvicorn started from wrong directory | Run from project root: `uvicorn backend.app.main:app` |
| All predictions return "Invalid Scan" | Model outputs are flat (ambiguous) | Check image quality and scan type selection |
| Low lung accuracy at inference | CLAHE mismatch or old training split | Ensure using current `train_lung.py`; re-train if needed |
| `ModuleNotFoundError: google.colab` | Running Colab script locally | Expected — Colab imports are guarded with `try/except` |
| Slow first prediction | TF JIT compilation | Normal; subsequent requests are fast |
