# System Workflow — Medical AI Cancer Detection

---

## 1. Brain Tumor Training Pipeline

```
Raw Dataset (Kaggle — masoudnickparvar/brain-tumor-mri-dataset)
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
       │
       ▼
Dataset Structure Validation
  validate_dataset() checks Training/ and Testing/ folders
  4 subfolders per split: glioma | meningioma | notumor | pituitary
  Raises RuntimeError if any folder missing
       │
       ▼
tf.data Pipeline
  image_dataset_from_directory()
    label_mode="int"  (sparse integer labels)
    class_names=["glioma","meningioma","notumor","pituitary"]
    image_size=(224,224), batch_size=32
  cast to float32 — NO divide by 255 (include_preprocessing=True)
  Augmentation (medium): flip, rotate±0.20, zoom±0.15, contrast, brightness, translate
       │
       ▼
Class Weights
  compute_class_weight("balanced") on training directory counts
       │
       ▼
Phase 1 — Head Training (backbone frozen)
  EfficientNetV2S (imagenet, include_preprocessing=True, include_top=False)
  Head: GAP → BN → Dropout(0.40) → Dense(512,relu,L2=1e-4) → BN → Dropout(0.20) → Softmax(4)
  Loss: SparseCategoricalCrossentropy
  LR: 1e-3 | Epochs: up to 15
  Callbacks: EarlyStopping(val_accuracy, patience=6)
             ReduceLROnPlateau(val_loss, factor=0.5, patience=3)
             ModelCheckpoint(output/brain/phase1_best.keras, val_accuracy)
       │
       ▼
Phase 2 — Backbone Fine-tuning
  Rebuild model with trainable_backbone=True
  Freeze first 200 layers; unfreeze layers 200+ (~half of EfficientNetV2S)
  Load Phase 1 weights from phase1_best.keras
  LR: 5e-6 | Epochs: up to p1_epochs_run + 60
  Callbacks: EarlyStopping(val_accuracy, patience=12)
             ModelCheckpoint → output/brain/brain_cancer_effnet.keras
       │
       ▼
Evaluation (val_ds = Testing/ directory)
  Classification report, confusion matrix, confidence distribution
  Calibration at thresholds: 0.50 → 0.95
       │
       ▼
Saved Artefacts
  output/brain/brain_cancer_effnet.keras   ← copy to backend/models/
  output/brain/metadata.json
  output/brain/confusion_matrix.png
  output/brain/training_curves.png
  output/brain/confidence_distribution.png
```

---

## 2. Lung Cancer Training Pipeline

```
Raw Dataset (Kaggle — hamdallak/the-iqothnccd-lung-cancer-dataset)
  kaggle datasets download -d hamdallak/the-iqothnccd-lung-cancer-dataset
       │
       ▼
Folder Discovery (fuzzy name matching)
  discover_class_folders() walks rglob("*")
  FOLDER_MAP handles "benign", "benign cases", "bengin cases", etc.
  Detects: benign | malignant | normal
       │
       ▼
Patient ID Inference  (prevent multi-slice leakage)
  infer_patient_id() tries:
    1. Parent subdirectory name
    2. Leading numeric prefix in filename (e.g. P001_slice04.jpg)
    3. Fallback: alphabetical batching, 20 slices per group
       │
       ▼
Three-Way Patient-Level Split  (GroupShuffleSplit × 2)
  Step 1: reserve 20% of patients as held-out test set
  Step 2: split remaining 80% → 85% train / 15% val
  Result: Train ≈ 68% │ Val ≈ 12% │ Test ≈ 20%
  Overlap assertion: assert not (train_groups & val_groups), etc.
       │
       ▼
CLAHE Preprocessing  (applied per channel before backbone)
  apply_clahe(): cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  load_and_preprocess(): BGR→RGB → CLAHE → resize(224,224) → float32
  CRITICAL: predict.py must apply identical CLAHE (see _apply_clahe)
       │
       ▼
tf.data Pipeline
  from_tensor_slices(paths, labels) → tf.py_function(load_and_preprocess)
  Augmentation (heavier, +GaussianNoise): flip, rotate, zoom, contrast, brightness, translate, noise
  Class weights computed on TRAIN set only (not val/test)
       │
       ▼
Phase 1 — Head Training (backbone frozen)
  EfficientNetV2S (imagenet, include_preprocessing=True, include_top=False)
  Head: GAP → BN → Dropout(0.50) → Dense(256,relu,L2=5e-4) → BN → Dropout(0.25) → Softmax(3)
  Loss: SparseFocalLoss(gamma=2.0, class_weights=class_weights)
        ← class weights baked into loss; do NOT also pass class_weight= to fit()
  LR: 5e-4 | Epochs: up to 20
  Callbacks: EarlyStopping(val_accuracy, patience=8)
             ReduceLROnPlateau(val_loss, factor=0.5, patience=4)
             ModelCheckpoint(output/lung/phase1_best.keras, val_accuracy)
       │
       ▼
Phase 2 — In-Place Backbone Fine-tuning
  backbone.trainable = True  (same object — no clear_session, no weight reload)
  Freeze first 250 layers of EfficientNetV2S
  LR: 2e-6 | Epochs: p1_epochs_run + 80 (correct epoch count, EarlyStopping-aware)
  Callbacks: EarlyStopping(val_accuracy, patience=15)
             ModelCheckpoint → output/lung/lung_cancer_final.keras
       │
       ▼
TTA Evaluation  (held-out test set — first contact)
  8× TTA augmentation per image, average probabilities
  Metrics: classification_report, confusion matrix, macro OvR AUC
  NOTE: test_ds was never used by any callback or EarlyStopping
       │
       ▼
Saved Artefacts
  output/lung/lung_cancer_final.keras   ← copy to backend/models/
  output/lung/metadata.json  (includes CLAHE requirement note)
  output/lung/confusion_matrix.png
  output/lung/training_curves.png
```

---

## 3. Colon Histopathology Training Pipeline

```
Raw Dataset (Kaggle — kmader/colorectal-histology-mnist / Kather 2016)
  kaggle datasets download -d kmader/colorectal-histology-mnist
       │
       ▼
Folder Discovery & Name Normalisation
  build_clean_structure() maps numbered folders:
    01_TUMOR → tumor, 02_STROMA → stroma, 03_COMPLEX → complex, ...
  Also handles alt names: "back"→empty, "norm"→mucosa, "tum"→tumor, etc.
  Copies all images to data/colon_clean/{class}/
  8 canonical classes (alphabetical):
    adipose | complex | debris | empty | lympho | mucosa | stroma | tumor
       │
       ▼
Spatial Group Assignment  (prevent WSI patch leakage)
  Kather 2016 does not expose patient IDs
  Approximation: sort images per class alphabetically,
  group consecutive 5 images into one group (batch_size=5)
  → ~125 groups per class (spatially coherent WSI regions kept together)
       │
       ▼
Group-Aware Train / Test Split  (GroupShuffleSplit)
  80% train / 20% test at group level
  Assertion: assert len(train_grp & test_grp) == 0
       │
       ▼
tf.data Pipeline (with MixUp)
  load_image(): tf.io.read_file → decode_image → resize(224,224) → float32
  make_train_pipeline():
    shuffle → load → batch(32) → augmentation → one_hot → apply_mixup(alpha=0.30)
    MixUp produces soft float labels → use CategoricalCrossentropy (not Sparse)
  make_val_pipeline():
    load → batch(32) → no augment, integer labels (for eval argmax)
       │
       ▼
Phase 1 — Head Training (backbone frozen)
  EfficientNetB2 (imagenet, include_preprocessing=True, include_top=False)
  Head: GAP → BN → Dropout(0.45) → Dense(256,relu,L2=3e-4) → BN → Dropout(0.225) → Softmax(8)
  Loss: CategoricalCrossentropy(label_smoothing=0.10)
  LR: 8e-4 | Epochs: up to 15
  Callbacks: EarlyStopping(val_accuracy, patience=6)
             ReduceLROnPlateau(val_loss, factor=0.5, patience=3)
             ModelCheckpoint(output/colon/phase1_best.keras, val_accuracy)
       │
       ▼
Phase 2 — Backbone Fine-tuning
  clear_session() → rebuild model → load_weights(phase1_best.keras)
  Freeze first 100 layers of EfficientNetB2 (~240 total)
  LR: 3e-6 | Epochs: p1_epochs_run + 70
  Callbacks: EarlyStopping(val_accuracy, patience=12)
             ModelCheckpoint → output/colon/colon_cancer_final.keras
  Note: uses rebuild approach (not in-place); works when layer names match
       │
       ▼
Evaluation (val_ds = test split, integer labels)
  Per-class F1 bar chart, confusion matrix, confidence distribution
  Calibration at thresholds: 0.50 → 0.90
  Expected range: 88–93% (matches Kather 2016 paper)
       │
       ▼
Saved Artefacts
  output/colon/colon_cancer_final.keras   ← copy to backend/models/
  output/colon/metadata.json
  output/colon/confusion_matrix.png
  output/colon/per_class_f1.png
  output/colon/training_curves.png
```

---

## 4. Inference Pipeline (per request)

```
HTTP POST /predict
  file: image bytes
  cancer_type: "brain" | "lung" | "colon"
       │
       ▼
main.py: content type check
  if content_type not in {image/jpeg, image/png, image/bmp, image/webp}
    → HTTP 415 Unsupported Media Type
       │
       ▼
Stage 1 — Input Validation  validate_input_bytes()
  size < 1 KB          → {"status":"rejected","reason":"invalid_input"}
  size > 20 MB         → {"status":"rejected","reason":"invalid_input"}
  PIL.Image.load()     → {"status":"rejected","reason":"invalid_input"}
       │
       ▼
preprocess_image(image_bytes, apply_clahe=False)
  PIL decode → convert("RGB") → resize(224,224, BILINEAR)
  → uint8 array → float32 (NO divide by 255)
  → np.expand_dims → shape: (1, 224, 224, 3)
       │
       ▼
Stage 2 — Domain Screening  screen_image_domain()
  Returns (is_suspicious: bool, reason_code: str, message: str).

  All signals exploit physical imaging properties (not protocol-dependent
  measurements), making them robust to CT acquisition variation, window/level
  settings, and JPEG compression artefacts.

  Signal 1 — Colour saturation ceiling (SATURATION_THRESHOLD in config.py):
    Convert preprocessed array to HSV; compute mean saturation (0–1).
    brain, lung: saturation > 0.15 → rejected, reason="non_medical_image"
    colon:       saturation > 0.85 → rejected, reason="non_medical_image"
    NOTE: colon ceiling raised from 0.50 to 0.85 — real H&E histopathology
    reaches mean saturation 0.55–0.70 due to eosin (pink) and haematoxylin
    (blue-purple) dyes.  0.85 is above any clinically observed H&E range.
    Catches logos, photographs, and normally-stained H&E histology sent to brain/lung.

  Signal 2 — Colour saturation floor (SATURATION_FLOOR in config.py):
    colon: saturation < 0.05 → rejected, reason="cross_modality"
    H&E process always introduces colour; near-grayscale images (CT/MRI)
    sent to the colon model are caught here.

  Signal 2b — Flat-colour image check, colon only (COLON_FLAT_IMAGE_THRESHOLDS):
    _check_colon_flat_image() uses RGB-derived saturation proxy (no cv2 call):
      sat_proxy = (max_channel - min_channel) / max_channel  per pixel
      mean_sat  = mean(sat_proxy)   sat_std = std(sat_proxy)
      pixel_std = std of grayscale (mean RGB / 255)
    Only fires when mean_sat > 0.35 (below this, lightly-coloured images are
    handled by Signal 2 and the quality gate).
    Rejection condition (AND logic — both must fail):
      sat_std   < 0.04  AND  pixel_std < 0.04
      → flat-colour graphic, not real tissue → rejected, reason="non_medical_image"
    Real H&E slides always have sat_std > 0.04 (varying staining density) or
    pixel_std > 0.04 (cell micro-texture), so valid tissue passes.

  Signal 3 — Intensity profile gate (MODALITY_INTENSITY_PROFILES in config.py):
    Saturation cannot distinguish brain MRI from lung CT (both near-grayscale).
    _compute_intensity_features() extracts four pixel statistics from the
    grayscale proxy (mean RGB / 255):
      dark_fraction  — fraction of pixels with normalised intensity < 0.20
      std            — overall pixel intensity standard deviation
      centre_mean    — mean intensity in the central [56:168, 56:168] block
      centre_std     — std of the central block
    _check_modality_compatibility() compares features against the expected
    profile for each modality (AND logic — both conditions must be exceeded):
      brain model: dark_fraction > 0.50 AND std > 0.32
                   → CT bimodal distribution (dark parenchyma + bright bone)
                   → rejected, reason="cross_modality"
      lung model:  centre_std < 0.08 AND centre_mean > 0.40 AND dark_fraction > 0.15
                   → brain MRI pattern (smooth bright centre, dark periphery)
                   → rejected, reason="cross_modality"
    Colon model not checked here — Signals 1/2/2b fully cover colon gating.

  Edge density (Laplacian variance) was evaluated and removed:
    It caused false rejections of valid lung CT scans — HRCT and nodule-dense CT
    can produce Laplacian variance values that overlap with histopathology ranges.
    Lightly-stained H&E that passes Signal 1 is handled downstream by Stage 3.
       │
       ▼
Stage 3 — Cancer Classification + Internal Quality Gate  classify_cancer()
  Lung CT only: apply_clahe=True
    → _apply_clahe(): cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) per channel
  cancer_model.predict(image_array) → softmax probs (num_classes,)  [internal only]
  _assess_prediction_quality(raw_probs, cancer_type):
    Thresholds are per-modality:
      brain / lung  → global:  CONF=0.35, MARGIN=0.07, ENTROPY_CEILING=0.95
      colon         → lenient: CONF=0.20, MARGIN=0.02, ENTROPY_CEILING=0.99
        (Kather 2016 low-cellularity classes — empty, stroma, adipose, debris —
         produce conf 20–30% and norm_H 0.96–0.98 on valid tissue; these are
         above the 8-class random baseline of 12.5% and must not be rejected)
    ENTROPY_FLOOR = 0.01 applies to ALL modalities (OOD commitment check):
      fires only at p₁ ≳ 99.9%; valid high-confidence predictions pass.

    confidence = max(probs)              < conf_threshold        → ambiguous
    margin = top_prob − 2nd_prob         < margin_threshold      → ambiguous
    norm_H = (−Σ p·log(p)) / ln(n)      > entropy_ceiling       → ambiguous (too flat)
    norm_H = (−Σ p·log(p)) / ln(n)      < ENTROPY_FLOOR (0.01)  → ambiguous (OOD peak)
    (normalising by ln(n_classes) makes entropy checks consistent
     across 3-class lung, 4-class brain, and 8-class colon models)
  Probabilities are used for all four criteria but are NOT returned in the response.
  If ambiguous:
    → predicted_class="Invalid Scan"
  Else:
    argmax → raw label → LABEL_MAPPING → human-readable name
  cancer_type unknown → {"status":"rejected","reason":"invalid_cancer_type"}
  model not loaded    → {"status":"error","reason":"model_unavailable"}
       │
       ▼
JSON Response (HTTP 200)
  {
    "status": "success",
    "cancer_type": "brain",
    "predicted_class": "Glioma"
  }
  If quality gate fails → predicted_class: "Invalid Scan" (status still "success")
```

---

## 5. Frontend ↔ Backend Request Flow

```
User (Browser)
    │ selects cancer type, uploads image
    ▼
Streamlit (port 8501)
    │ st.file_uploader → bytes
    │ requests.post("http://127.0.0.1:8000/predict",
    │               files={"file": bytes},
    │               data={"cancer_type": selected_type})
    ▼
FastAPI (port 8000)
    │ UploadFile → await file.read() → bytes
    │ runs full 3-stage inference pipeline
    │ returns PredictionResponse JSON
    ▼
Streamlit
    │ response.json() → status check
    │ "success"  → display prediction card, probability bar chart, medical context
    │ "rejected" → display error with reason
    │ "error"    → display model-unavailable message
    ▼
User sees result
```

---

## 6. Config.py — Single Source of Truth

All thresholds, paths, and class labels live in `backend/app/config.py`.
Never hard-code these values in logic files.

| Constant | Purpose |
|----------|---------|
| `CONF_THRESHOLD` | global min softmax top-class prob (0.35); below this → ambiguous |
| `MARGIN_THRESHOLD` | global min top-2 prob gap (0.07); below this → ambiguous. Set to 0.07 (not 0.10) to allow valid notumor predictions (typically 7–12% margin due to class imbalance in the training dataset) |
| `ENTROPY_THRESHOLD` | global max *normalised* entropy H/H_max (0.95); above this → ambiguous (too flat) |
| `ENTROPY_FLOOR` | min *normalised* entropy H/H_max (0.01, all modalities); below this → ambiguous (OOD over-concentration). Set to 0.01 (not 0.05) to allow valid 99–99.8% confidence predictions |
| `COLON_QUALITY_THRESHOLDS` | colon-specific quality gate overrides: conf=0.20, margin=0.02, entropy_ceiling=0.99. Low-cellularity tissue classes (empty, stroma, adipose, debris) produce conf 20–30% and norm_H 0.96–0.98 on valid tissue — above the 8-class random baseline but outside the global thresholds. ENTROPY_FLOOR is not overridden. |
| `SATURATION_THRESHOLD` | per-modality mean HSV saturation ceiling; above this → `non_medical_image` (brain/lung: 0.15; colon: 0.85 — real H&E reaches 0.55–0.70) |
| `SATURATION_FLOOR` | per-modality minimum HSV saturation; below this → `cross_modality` (colon only: CT/MRI are near-grayscale) |
| `COLON_FLAT_IMAGE_THRESHOLDS` | RGB-derived texture thresholds for Signal 2b: suspicious_sat_min=0.35, sat_std_min=0.04, pixel_std_min=0.04; flat-colour colon graphics rejected when both std values are below their minimums |
| `MODALITY_INTENSITY_PROFILES` | per-modality pixel intensity thresholds for Signal 3; brain checks CT bimodal signature; lung checks brain-MRI smooth-centre signature |
| `CLASS_LABELS` | raw training label order per model (must match Keras alphabetical sort) |
| `LABEL_MAPPING` | raw label → human-readable display name |
| `MODEL_PATHS` | absolute paths to .keras files (resolved relative to config.py) |
| `IMAGE_SIZE` | (224, 224) — shared across all models |

---

## 7. Key Invariants

1. **Run uvicorn from the project root** — `config.py` resolves model paths relative
   to its own file location. Starting from `backend/` shifts all paths by one directory.

2. **Raw [0, 255] float32 only** — all models use `include_preprocessing=True`.
   Never divide pixel values by 255 before passing to any model.

3. **CLAHE for lung CT only** — `train_lung.py` applies CLAHE; `predict.py` mirrors
   this exactly. Brain and colon models were not trained with CLAHE.

4. **Patient-level splits for lung** — `train_lung.py` uses two rounds of
   `GroupShuffleSplit` at patient level. Image-level splits on multi-slice CT data
   produce contaminated test sets and inflated accuracy.

5. **Colon labels must be CategoricalCrossentropy** — MixUp produces soft float
   labels. `SparseCategoricalCrossentropy` (which expects integer indices) cannot
   be used for the colon model.
