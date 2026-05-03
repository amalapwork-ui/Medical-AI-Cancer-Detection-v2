# Operation Manual — Medical AI Cancer Detection System

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB (excl. models) | 1 GB |
| GPU | Not required (inference runs on CPU) | CUDA GPU for faster inference |

> tensorflow-cpu 2.21.0 runs on CPU. Inference takes ~0.1–0.5 s per image after
> JIT warmup (~2–5 s on first request per model per server restart).

---

## 2. First-Time Setup

### 2.1 Create Virtual Environment

```bash
python -m venv myvenv

# Windows
myvenv\Scripts\activate

# macOS / Linux
source myvenv/bin/activate
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages: FastAPI, Uvicorn, tensorflow-cpu 2.21.0, Pillow, opencv-python-headless
(required for CLAHE on lung CT images), Streamlit, scikit-learn.

**TensorFlow version note:** `requirements.txt` pins `tensorflow-cpu==2.21.0`. Use
`tensorflow-cpu`, **not** `tensorflow` — the GPU variant loads `nvcuda.dll` at
DLL-load time on Windows and hangs indefinitely when CUDA is absent. Do not
upgrade without re-validating EfficientNetV2S model loading and
`include_preprocessing=True` behaviour.

### 2.3 Place Model Files

Trained `.keras` files are in `backend/models/` and tracked in the repository.

```
backend/models/
├── brain_cancer_effnet.keras
├── lung_cancer_final.keras
└── colon_cancer_final.keras
```

The server starts with missing models — `GET /health` reports which are absent
and inference proceeds in degraded mode (missing classifier = error response
for that cancer type). Quality-gate rejection (ambiguous predictions) requires
no additional model files — it is computed from the classifier's own output.

---

## 3. Starting the System

Two terminals are required (backend and frontend are separate processes).

### Terminal 1 — Backend (FastAPI)

```bash
# Activate venv first, then:
uvicorn backend.app.main:app --reload
```

**Must be run from the project root**, not from `backend/`. The `config.py`
module resolves all model paths relative to its own file location; running from
the wrong directory produces `FileNotFoundError` on model paths.

Expected startup output:
```
INFO: Server starting — loading models …
INFO: Loaded 'brain' ✓
INFO: Loaded 'lung' ✓
INFO: Loaded 'colon' ✓
INFO: Ready.  Loaded: ['brain', 'lung', 'colon']  Missing: []
INFO: Uvicorn running on http://127.0.0.1:8000
```

Endpoints:
- `GET  /health`  — model status
- `POST /predict` — inference
- `GET  /docs`    — interactive API documentation (Swagger UI)

### Terminal 2 — Frontend (Streamlit)

```bash
streamlit run frontend/streamlit_app.py
```

Opens automatically at `http://localhost:8501`.

---

## 4. API Reference

### GET /health

```json
{
  "status": "ok",
  "models_loaded": ["brain", "lung", "colon"],
  "models_missing": []
}
```

`status` is `"degraded"` if any model failed to load or was missing.

### POST /predict

**Request:** `multipart/form-data`

| Field | Type | Valid values |
|-------|------|--------------|
| `file` | image file | JPEG, PNG, BMP, WebP (≤ 20 MB, ≥ 1 KB) |
| `cancer_type` | string | `brain` \| `lung` \| `colon` (case-insensitive) |

**Success response:**

```json
{
  "status": "success",
  "cancer_type": "lung",
  "predicted_class": "Malignant Lung Cancer"
}
```

When the model cannot produce a reliable prediction, `predicted_class` is `"Invalid Scan"`:

```json
{
  "status": "success",
  "cancer_type": "brain",
  "predicted_class": "Invalid Scan"
}
```

**How the internal quality gate works (hidden from API response):**

The system computes the full softmax probability distribution on every request. These probabilities are used internally by four criteria before the predicted class is finalised. Thresholds are per-modality:

| Internal check | Brain / Lung | Colon | Purpose |
|----------------|-------------|-------|---------|
| Confidence floor | ≥ 0.35 | ≥ 0.20 | Rejects near-random outputs |
| Margin | ≥ 0.07 | ≥ 0.02 | Rejects near-ties between the top-2 classes |
| Entropy ceiling | H/H_max ≤ 0.95 | H/H_max ≤ 0.99 | Rejects flat, uncertain distributions |
| Entropy floor | H/H_max ≥ 0.01 | H/H_max ≥ 0.01 | Rejects true OOD over-commitment (p₁ ≳ 99.9%) |

If any check fails, `predicted_class` is set to `"Invalid Scan"`. The raw probabilities and intermediate values are never returned to the caller — only the final class name is exposed.

**Why colon uses lenient thresholds:** The Kather 2016 8-class model covers low-cellularity tissue classes (empty background tiles, sparse stroma, adipose tissue, cellular debris) that produce genuinely weaker softmax signals than tumour classes. These valid inputs return conf 20–30% and norm_H 0.96–0.98 — above the 8-class random baseline of 12.5% but below the global thresholds calibrated for 3- and 4-class models. The colon-specific thresholds (conf ≥ 0.20, margin ≥ 0.02, entropy ≤ 0.99) accept these uncertain-but-valid tissue predictions. Truly random outputs (conf < 20%) and near-uniform distributions (norm_H > 0.99) are still rejected.

**Why the margin threshold is 0.07 (global), not 0.10:** The brain model's `notumor` class has ~2× fewer training examples (~395 vs ~826 per tumour class) in the masoudnickparvar dataset. Detecting the absence of a tumour is also inherently harder than detecting its presence. Valid notumor predictions typically have margins of 7–12%; a threshold of 0.10 incorrectly rejected these. 0.07 still filters true near-ties while accepting small-but-decisive notumor margins.

**Why the entropy floor is 0.01, not higher:** A well-trained EfficientNetV2S model on clear medical images regularly produces 99–99.7% confidence on highly distinctive classes (pituitary adenomas, normal brain scans, normal lung). A floor set at 0.05 would incorrectly reject these as OOD. The 0.01 threshold only fires when essentially all probability mass collapses onto one class (≳ 99.9%), which trained models do not produce on valid medical images.

**Rejection response (invalid input):**

```json
{
  "status": "rejected",
  "reason": "invalid_input",
  "message": "File is too small (500 bytes). Upload a valid medical image."
}
```

**Rejection response (non-medical image — colourful logo or photograph):**

```json
{
  "status": "rejected",
  "reason": "non_medical_image",
  "message": "Image colour saturation (0.42) exceeds the expected range for brain imaging (threshold: 0.15). Upload a genuine medical scan."
}
```

**Rejection response (cross-modality — medical image sent to wrong model):**

```json
{
  "status": "rejected",
  "reason": "cross_modality",
  "message": "Image colour saturation (0.02) is below the expected range for colon imaging (minimum: 0.05). Colon histopathology scans show H&E staining colour. If you submitted a CT or MRI scan, select the correct cancer type."
}
```

Or for a lung CT submitted to the brain model (Signal 3):

```json
{
  "status": "rejected",
  "reason": "cross_modality",
  "message": "Image intensity profile (dark fraction 0.61, pixel std 0.43) matches a CT scan rather than a brain MRI. Brain MRI has a more concentrated intensity distribution with dark fraction typically below 0.50. If this is a lung CT scan, select 'Lung' as the cancer type."
}
```

Or for a brain MRI submitted to the lung model (Signal 3):

```json
{
  "status": "rejected",
  "reason": "cross_modality",
  "message": "Image centre region is uniformly bright (centre std 0.00, centre mean 0.51) with dark peripheral background (dark fraction 0.33), which is characteristic of brain MRI tissue rather than a lung CT scan. Lung CT always shows dark parenchyma within the central image region. If this is a brain MRI, select 'Brain' as the cancer type."
}
```

| `reason` | Cause |
|----------|-------|
| `invalid_input` | File < 1 KB, > 20 MB, corrupted, or unrecognised format |
| `non_medical_image` | Colour saturation exceeds the per-modality ceiling — likely a logo, photograph, or screenshot |
| `cross_modality` | Medical image from the wrong modality — CT/MRI sent to colon model, or histopathology sent to brain/lung model |
| `invalid_cancer_type` | `cancer_type` not in `brain \| lung \| colon` |

**Error response (model not loaded):**

```json
{
  "status": "error",
  "reason": "model_unavailable",
  "message": "Model for 'brain' is not loaded. Check /health."
}
```

**cURL example:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@brain_scan.jpg;type=image/jpeg" \
  -F "cancer_type=brain"
```

---

## 5. Output Classes

**Brain** (4 classes — MRI):
| Raw label | Display name |
|-----------|-------------|
| glioma | Glioma |
| meningioma | Meningioma |
| notumor | No Tumor |
| pituitary | Pituitary Tumor |

**Lung** (3 classes — CT, IQ-OTH/NCCD dataset):
| Raw label | Display name |
|-----------|-------------|
| benign | Benign Lung Lesion |
| malignant | Malignant Lung Cancer |
| normal | Normal Lung |

**Colon** (8 classes — histopathology, Kather 2016 dataset):
| Raw label | Display name |
|-----------|-------------|
| adipose | Adipose Tissue |
| complex | Complex Glandular Epithelium |
| debris | Cellular Debris |
| empty | Background / Empty |
| lympho | Lymphocytic Infiltrate |
| mucosa | Normal Mucosa |
| stroma | Cancer-Associated Stroma |
| tumor | Colorectal Adenocarcinoma |

---

## 6. Running Tests

Tests do not require GPU, model files, or a running server.

```bash
# All tests
pytest tests/ -v

# Unit tests only (inference logic with mocked TF)
pytest tests/test_inference.py -v

# API integration tests only (FastAPI TestClient, mocked models)
pytest tests/test_api.py -v

# Model spec tests (output shapes, class counts)
pytest tests/test_models.py -v

# Modality gate tests (Signal 3: intensity profile checks)
pytest tests/test_inference.py::TestModalityIntensityGate -v

# Single test by name
pytest tests/test_inference.py::TestFullPipeline::test_valid_brain_scan -v
```

---

## 7. Training Models

All training scripts are designed for Google Colab (GPU recommended). Each
script downloads its own dataset via the Kaggle API and saves the final model
to `output/<model>/`. Copy the `.keras` file to `backend/models/` when done.

### 7.1 Train the Brain Tumor Model

1. Open `training/train_brain.py` in Google Colab
2. Run Cell 2 — upload `kaggle.json` when prompted
3. Run all remaining cells (dataset downloads, two-phase training, evaluation)
4. Cell 13 downloads `brain_cancer_effnet.keras` automatically
5. Copy to `backend/models/brain_cancer_effnet.keras`

Dataset: `masoudnickparvar/brain-tumor-mri-dataset`
Expected accuracy: ~91% on the provided Testing/ split
Training time: ~30–60 min on Colab T4 GPU

### 7.2 Train the Lung Cancer Model

1. Open `training/train_lung.py` in Google Colab
2. Run Cell 2 — upload `kaggle.json`
3. Run all cells — script downloads IQ-OTH/NCCD, builds patient-level splits,
   trains with SparseFocalLoss, evaluates with TTA
4. Cell 18 downloads `lung_cancer_final.keras`
5. Copy to `backend/models/lung_cancer_final.keras`

Dataset: `hamdallak/the-iqothnccd-lung-cancer-dataset`
Expected accuracy: ~82–90% (TTA, held-out patient-level test set)
Training time: ~40–80 min on Colab T4 GPU

> **Note:** The lung model applies CLAHE preprocessing during training. The
> inference pipeline in `predict.py` applies identical CLAHE for lung CT images.
> If you modify CLAHE parameters in the training script, update `_apply_clahe()`
> in `backend/app/predict.py` to match.

### 7.3 Train the Colon Tissue Classifier

1. Open `training/train_colon.py` in Google Colab
2. Run Cell 2 — upload `kaggle.json`
3. Run all cells — downloads Kather 2016 dataset, discovers and renames class
   folders, builds spatial group-aware split, trains with MixUp + label smoothing
4. Cell 17 downloads `colon_cancer_final.keras`
5. Copy to `backend/models/colon_cancer_final.keras`

Dataset: `kmader/colorectal-histology-mnist` (Kather 2016 histopathology)
Expected accuracy: ~88–93%
Training time: ~30–50 min on Colab T4 GPU

> **Colon class labels:** The model uses 8 tissue classes from Kather 2016
> (adipose, complex, debris, empty, lympho, mucosa, stroma, tumor).
> These are already set in `backend/app/config.py → CLASS_LABELS["colon"]`.

---

## 8. Adding a New Dataset

To retrain a model on a different dataset:

1. **Update the training script** — change the Kaggle download command and
   class folder paths in the appropriate `train_<organ>.py` file.

2. **Update `CLASS_LABELS`** in `backend/app/config.py` — the list must match
   the alphabetical order Keras assigns when building a dataset from directories.

3. **Update `LABEL_MAPPING`** in `backend/app/config.py` — add human-readable
   display names for any new raw class labels.

4. **Retrain** using the updated script.

5. **Test** with `pytest tests/ -v` after placing the new model in `backend/models/`.

---

## 9. Troubleshooting

### Backend fails to find model files

```
WARNING: Model file not found — skipping: …/backend/models/brain_cancer_effnet.keras
```

**Fix:** Run `uvicorn` from the project root:
```bash
# Wrong
cd backend && uvicorn app.main:app

# Correct (from project root)
uvicorn backend.app.main:app --reload
```

---

### Cross-modality images not being rejected

The system uses four signals to detect wrong-modality submissions. All exploit physical imaging properties and are robust to CT protocol variation, window settings, and JPEG compression.

**Signal 1 — Saturation ceiling (per-modality):** Normally stained H&E slides (saturation 0.15–0.35) sent to the brain or lung model are caught here. For the colon model the ceiling is 0.85 — real H&E histopathology reaches mean saturation 0.55–0.70, so the previous ceiling of 0.50 was incorrectly rejecting valid tissue images. Images above 0.85 return `reason: "non_medical_image"`.

**Signal 2 — Saturation floor (colon only):** A CT or MRI scan (near-grayscale, saturation ≈ 0.01–0.03) sent to the colon model fails the 0.05 floor and returns `reason: "cross_modality"`. If a grayscale CT is passing through, check that `cancer_type=colon` is being sent — this signal only applies to the colon route.

**Signal 2b — Flat-colour check (colon only):** Solid-colour graphics (logos, swatches) that have enough colour to pass Signals 1 and 2 are caught here. Real H&E slides always have spatial variation in staining density (sat_std > 0.04) or cell micro-texture (pixel_std > 0.04). A solid uniform image fails both. Both must fail simultaneously (AND logic) to avoid false rejections of uniformly-stained tissue. Returns `reason: "non_medical_image"`.

**Signal 3 — Intensity profile (brain and lung models):** Brain MRI and lung CT both produce near-grayscale images (similar saturation), so Signals 1 and 2 cannot separate them. Signal 3 uses the *shape* of the pixel intensity distribution:

- **Lung CT → brain model:** Lung CT has a bimodal histogram — very dark lung parenchyma (0–50/255) and bright mediastinum/bone (150–230/255). This produces high `dark_fraction` (> 0.50) AND high pixel `std` (> 0.32). A brain MRI does not produce this bimodal signature. Both conditions must be exceeded (AND logic) to trigger rejection.

- **Brain MRI → lung model:** Brain MRI has smooth uniform brain tissue in the central image region (low `centre_std` < 0.08, elevated `centre_mean` > 0.40) with dark peripheral background (`dark_fraction` > 0.15). Lung CT always has dark parenchyma extending into the centre, making the centre heterogeneous. All three conditions must be exceeded (AND logic).

**Lightly-stained histopathology → brain/lung (not rejected by domain screen):** If an unusually pale H&E slide passes Signal 1, the softmax quality gate handles it: the brain/lung model produces either a flat or anomalously peaked distribution, returning `prediction_status: "ambiguous"`.

**Why edge density (Laplacian variance) is NOT used as a signal:** Edge density was evaluated and removed because it caused false rejections of valid lung CT scans. HRCT, nodule-dense scans, and narrow-window JPEG exports can produce Laplacian variance values that overlap with histopathology. Intensity distribution shape (Signal 3) operates at a coarser scale and is stable across CT protocols.

**Remaining limitation:** Greyscale non-medical images (black-and-white photos, greyscale screenshots) pass Signal 1 and may not produce a CT or MRI intensity profile. The softmax quality gate is the fallback.

---

### Non-medical image not being rejected

The system uses three layers to screen non-medical inputs:

**Signal 1 — Colour saturation ceiling (before model)**: Brain MRI and lung CT are near-grayscale (threshold 0.15). Colon H&E histology allows high colour (threshold 0.85 — real H&E reaches 0.55–0.70 mean saturation). Colourful logos, screenshots, and photographs should be caught here and return `reason: "non_medical_image"`.

If a colourful image is passing through, check whether the correct `cancer_type` is being sent — the saturation threshold is modality-specific.

**Signal 2b — Flat-colour check (colon model only, before model):** Solid-colour graphics that fall within the saturation window (0.35–0.85) are caught by `_check_colon_flat_image()`, which checks for spatial variation in saturation and pixel intensity. A real H&E slide always has one or both above 0.04; a logo or swatch does not.

**Entropy floor (after model)**: If a greyscale non-medical image passes the saturation check, the softmax distribution quality gate applies an entropy floor (H/H_max ≥ 0.01). A model that assigns > 99.9% to one class fails this check and returns `predicted_class: "Invalid Scan"`.

**Note on edge density:** Laplacian variance was previously used as an additional
pre-screen signal but was removed after it caused false rejections of valid
lung CT scans (HRCT and high-contrast exports can produce edge density values
that overlap with histopathology). Lightly-stained histopathology that passes
Signal 1 is now handled by the entropy-based quality gate.

**Known limitation**: A greyscale non-medical image that produces a moderate
(not extreme) softmax output may pass both layers. This is a fundamental
constraint of softmax-based classifiers without a dedicated OOD model.

---

### Valid scans marked "Invalid Scan" (ambiguous)

The quality gate thresholds are per-modality. Brain/lung: confidence ≥ 0.35,
margin ≥ 0.07, entropy ceiling H/H_max ≤ 0.95, entropy floor ≥ 0.01.
Colon: confidence ≥ 0.20, margin ≥ 0.02, entropy ceiling ≤ 0.99
(low-cellularity tissue classes produce conf 20–30% by design; see the
quality gate note above). The entropy floor (0.01) is the same for all
modalities and fires only at p₁ ≳ 99.9%.

Real-world CT scans returning 50–65% top-class confidence should pass the
brain/lung gate. A 98% confidence prediction also passes (norm_H ≈ 0.086
for a 4-class model with that distribution).

If valid images are being rejected, the most likely cause is using an
older model trained without focal loss that produces lower-calibrated
outputs. Retrain with the current training scripts.

---

### Low lung cancer accuracy (getting ~60–70%, expect ~82–90%)

Two previously documented bugs can cause this. If you trained with an older
version of the script, retrain with the current `train_lung.py`:

- **CLAHE mismatch:** Training applied CLAHE; older inference did not.
  The current `predict.py` applies CLAHE for lung images (`apply_clahe=True`).

- **Data leakage in split:** An older split aliased the test set as the
  validation set, causing EarlyStopping and ModelCheckpoint to optimise on
  the test set. The current script uses a proper 3-way patient-level split.

---

### `ModuleNotFoundError: No module named 'google.colab'`

Training scripts detect whether they are running in Colab or locally using
a `try/except ImportError`. If this error appears:
- Check you are using the current training scripts (the `_IN_COLAB` guard is
  present in `train_lung.py` and `train_validator.py`)
- `train_brain.py` and `train_colon.py` have bare `from google.colab import files`
  in some cells — run those scripts in Colab, or comment out those cells locally.

---

### Slow first prediction (2–10 s)

TensorFlow performs JIT compilation on the first `model.predict()` call per
model per server restart. This is expected. Subsequent requests complete in
~0.1–0.5 s on CPU.

---

### Oversized model loading (slow startup)

All three models load sequentially at startup. On a CPU-only machine, this
takes approximately 10–30 seconds total. The server is not ready for requests
until startup is complete (uvicorn will log `Ready.` when done).

---

### Test failures related to `cv2` import

The `test_api.py` file stubs `cv2` so tests run without `opencv-python-headless`
installed. If you run `test_inference.py` directly and get a `cv2` ImportError,
either install opencv or run via `pytest tests/ -v` (which imports test_api.py
first, injecting the cv2 stub into `sys.modules`). Alternatively, install the
full requirements: `pip install -r requirements.txt`.
