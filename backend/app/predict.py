"""
Production inference pipeline — Medical AI Cancer Detection.

Flow:
    1. validate_input_bytes()        — reject corrupted / oversized / non-image files
    2. preprocess_image()            — decode → RGB → (CLAHE for lung) → resize → float32
    3. screen_image_domain()         — three-signal domain check (no extra model):
         Signal 1: saturation ceiling  — catches logos/photos and high-saturation H&E
         Signal 2: saturation floor    — catches CT/MRI sent to colon model
         Signal 3: intensity profile   — catches lung CT sent to brain model, and
                                         brain MRI sent to lung model
    4. classify_cancer()             — route to correct cancer model
    5. _assess_prediction_quality()  — confidence / margin / entropy gate (floor + ceiling)
    6. build response                — structured JSON

Key design decisions:
  - No separate validator or OOD model. Domain screening uses three signals that
    exploit per-modality physical properties of medical imaging, all computed
    from pixel statistics with no extra model required:
      Signal 1 (saturation ceiling): MRI/CT are near-grayscale; logos/photos and
        normally-stained H&E histopathology are much more saturated.
      Signal 2 (saturation floor): H&E staining always introduces colour; a
        near-grayscale image sent to the colon model is almost certainly CT/MRI.
      Signal 3 (intensity profile): lung CT has a bimodal pixel distribution
        (dark parenchyma + bright mediastinum); brain MRI has a smooth bright
        central region with dark only at the periphery.  These distribution
        shapes are captured by dark_fraction + pixel std (for brain model) and
        centre_std + centre_mean (for lung model).
    Quality is then assessed from the cancer model's own output distribution
    using four criteria: confidence, prediction margin, normalised Shannon
    entropy ceiling (H/H_max), and entropy floor.  Normalising entropy by
    ln(n_classes) makes thresholds consistent across 3-class (lung), 4-class
    (brain), and 8-class (colon) models.
  - Models are loaded once at startup into a module-level registry dict.
  - Preprocessing is 100% consistent between training and inference:
      All models: raw [0, 255] float32 → EfficientNetV2S.include_preprocessing=True
      Lung CT only: CLAHE applied before the backbone, matching train_lung.py
  - All thresholds live in config.py — never hard-coded here.
"""

import io
import logging
import os
from typing import Optional

# Must be set before TF is imported.
# TF's CUDA scanner runs inside its C++ DLL loader; for CUDA_VISIBLE_DEVICES=-1
# to take effect it must be in the OS environment when the process starts.
# These are last-resort fallbacks — the correct place is run_backend.bat.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from .config import (
    CLASS_LABELS,
    COLON_EXCLUDED_CLASSES,
    COLON_FLAT_IMAGE_THRESHOLDS,
    COLON_QUALITY_THRESHOLDS,
    CONF_THRESHOLD,
    ENTROPY_FLOOR,
    ENTROPY_THRESHOLD,
    IMAGE_SIZE,
    LABEL_MAPPING,
    MARGIN_THRESHOLD,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    MODALITY_INTENSITY_PROFILES,
    MODEL_PATHS,
    SATURATION_FLOOR,
    SATURATION_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

_models: dict[str, tf.keras.Model] = {}


def load_models() -> None:
    """
    Load all cancer models into the in-process registry.
    Called once from the FastAPI lifespan event — never per-request.
    Missing model files are skipped; the /health endpoint reports them.
    """
    global _models
    logger.info("[load_models] TF version: %s", getattr(tf, "__version__", "stub"))
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            logger.warning("[load_models] NOT FOUND — skipping '%s': %s", name, path)
            continue
        try:
            logger.info("[load_models] Loading '%s' from %s …", name, path)
            # compile=False skips loss/optimizer reconstruction.
            # The lung model uses a custom SparseFocalLoss that is not
            # registered in this process; we never call model.compile()
            # or model.fit() at inference time, so this is safe.
            _models[name] = tf.keras.models.load_model(str(path), compile=False)
            logger.info("[load_models] Loaded '%s' ✓", name)
        except Exception as exc:
            logger.error("[load_models] FAILED to load '%s': %s", name, exc)

    loaded  = list(_models.keys())
    missing = [n for n in MODEL_PATHS if n not in _models]
    logger.info("[load_models] Complete — loaded=%s  missing=%s", loaded, missing)


def loaded_models() -> list[str]:
    return list(_models.keys())


def missing_models() -> list[str]:
    return [n for n in MODEL_PATHS if n not in _models]


def _get_model(name: str) -> Optional[tf.keras.Model]:
    return _models.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_input_bytes(image_bytes: bytes) -> None:
    """
    Lightweight validation before any model inference.
    Raises ValueError with a user-facing message on failure.
    """
    size = len(image_bytes)

    if size < MIN_FILE_SIZE_BYTES:
        raise ValueError(
            f"File is too small ({size} bytes). "
            "Upload a valid medical image."
        )

    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit. "
            "Please compress or resize the image."
        )

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # full decode — verify() is too lenient with truncated JPEGs
    except UnidentifiedImageError:
        raise ValueError("The uploaded file is not a recognised image format.")
    except Exception as exc:
        raise ValueError(f"Image file appears corrupted: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE — lung CT preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _apply_clahe(arr: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE per channel to a uint8 (H, W, 3) array.
    Replicates train_lung.py: clipLimit=2.0, tileGridSize=(8,8).
    Must only be applied to lung CT — the lung model was trained on CLAHE-enhanced
    slices; brain and colon models were not.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = [clahe.apply(arr[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, apply_clahe: bool = False) -> np.ndarray:
    """
    Decode → RGB → (optional CLAHE) → resize (bilinear) → float32 [0, 255].

    EfficientNetV2S was saved with include_preprocessing=True, so the backbone
    applies its own pixel-value normalisation internally.
    Do NOT divide by 255 — that would break the backbone's expectations.

    Returns ndarray of shape (1, 224, 224, 3), dtype float32, values in [0, 255].
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    if apply_clahe:
        arr = _apply_clahe(arr)
    return np.expand_dims(arr.astype(np.float32), axis=0)  # (1, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Modality intensity profiling  (Signal 3 helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_intensity_features(image_array: np.ndarray) -> dict[str, float]:
    """
    Extract pixel intensity statistics for cross-modality detection.

    All features are computed on the grayscale equivalent (mean of RGB / 255),
    which is valid for near-grayscale medical images (MRI and CT).  The colon
    model is not checked by this function (_check_modality_compatibility returns
    False immediately for colon), so histopathology inputs passing the saturation
    ceiling do not conflict with these intensity statistics.

    Features returned:
        dark_fraction  — fraction of pixels with normalised intensity < 0.20.
                         High in lung CT (lung parenchyma + circular FOV background).
        std            — pixel intensity standard deviation (0–1 range).
                         High in CT due to bimodal distribution; moderate in MRI.
        centre_mean    — mean intensity in the central 112×112 block [56:168, 56:168].
                         Elevated in brain MRI (uniform grey/white matter).
        centre_std     — std of the central 112×112 block.
                         Low in brain MRI (smooth brain tissue); higher in lung CT
                         (dark parenchyma extends into the centre alongside the
                         bright mediastinum, creating within-block heterogeneity).
    """
    img  = image_array[0].astype(np.float32)
    gray = np.mean(img, axis=-1) / 255.0   # (224, 224), values in [0, 1]
    flat = gray.flatten()

    dark_fraction = float(np.mean(flat < 0.20))
    std           = float(np.std(flat))

    centre      = gray[56:168, 56:168].flatten()   # central 50% of 224×224
    centre_mean = float(np.mean(centre))
    centre_std  = float(np.std(centre))

    return {
        "dark_fraction": dark_fraction,
        "std"          : std,
        "centre_mean"  : centre_mean,
        "centre_std"   : centre_std,
    }


def _check_modality_compatibility(
    features: dict[str, float], cancer_type: str
) -> tuple[bool, str]:
    """
    Compare intensity features against the expected profile for cancer_type.
    Returns (is_mismatch, rejection_message).

    Each check uses AND logic — both conditions must be satisfied for rejection.
    This conservative approach prevents single-metric edge cases (e.g. a high-
    contrast MRI with unusually dark background) from causing false rejections.

    Modalities handled:
        brain — detects CT-bimodal signature (high dark fraction + high std).
        lung  — detects brain-MRI signature (smooth uniform bright centre).
        colon — not checked here; saturation signals fully cover colon gating.
    """
    profile = MODALITY_INTENSITY_PROFILES.get(cancer_type)
    if profile is None:
        return False, ""

    dark   = features["dark_fraction"]
    std    = features["std"]
    c_mean = features["centre_mean"]
    c_std  = features["centre_std"]

    if cancer_type == "brain":
        if dark > profile["ct_dark_fraction_min"] and std > profile["ct_std_min"]:
            return True, (
                f"Image intensity profile (dark fraction {dark:.2f}, "
                f"pixel std {std:.2f}) matches a CT scan rather than a brain MRI. "
                "Brain MRI has a more concentrated intensity distribution with "
                f"dark fraction typically below {profile['ct_dark_fraction_min']:.2f}. "
                "If this is a lung CT scan, select 'Lung' as the cancer type."
            )

    elif cancer_type == "lung":
        if (
            c_std  < profile["brain_centre_std_max"]
            and c_mean > profile["brain_centre_mean_min"]
            and dark  > profile["brain_dark_fraction_min"]
        ):
            return True, (
                f"Image centre region is uniformly bright (centre std {c_std:.2f}, "
                f"centre mean {c_mean:.2f}) with dark peripheral background "
                f"(dark fraction {dark:.2f}), which is characteristic of brain MRI "
                "tissue rather than a lung CT scan. Lung CT always shows dark "
                "parenchyma within the central image region. "
                "If this is a brain MRI, select 'Brain' as the cancer type."
            )

    return False, ""


def _check_colon_flat_image(image_array: np.ndarray) -> tuple[bool, str]:
    """
    Signal 2b — detect flat-colour non-medical images submitted to the colon model.

    Real H&E histopathology has two properties that solid-colour graphics lack:
      1. Spatial variation in staining density   → sat_std  > threshold
      2. Micro-texture (cells, glands, nuclei)   → pixel_std > threshold

    Both must fall below their thresholds (AND logic) to trigger rejection.
    Conservative by design: uniformly-stained tissue patches (e.g. Kather 2016
    "empty" background tiles, mucosa) either have mean_sat below the
    suspicious_sat_min gate or sufficient pixel_std from residual staining.

    Uses an RGB-derived saturation proxy instead of cv2 HSV so this check is
    independent of the colour-space conversion used in Signals 1 and 2.

    Returns:
        (is_suspicious: bool, message: str)
    """
    thresholds = COLON_FLAT_IMAGE_THRESHOLDS
    img_rgb    = image_array[0] / 255.0     # (H, W, 3), float32 in [0, 1]

    # RGB-derived saturation proxy: (max_channel - min_channel) / max_channel
    rgb_max   = np.max(img_rgb, axis=-1)    # (H, W)
    rgb_min   = np.min(img_rgb, axis=-1)
    sat_proxy = np.where(rgb_max > 1e-6,
                         (rgb_max - rgb_min) / (rgb_max + 1e-8),
                         0.0)

    mean_sat = float(np.mean(sat_proxy))

    # For lightly-coloured images the saturation floor (Signal 2) and the
    # quality gate are sufficient; skip the texture check.
    if mean_sat < thresholds["suspicious_sat_min"]:
        return False, ""

    sat_std   = float(np.std(sat_proxy))
    pixel_std = float(np.std(np.mean(img_rgb, axis=-1)))   # grayscale std

    if sat_std < thresholds["sat_std_min"] and pixel_std < thresholds["pixel_std_min"]:
        return True, (
            f"Image has high colour saturation (RGB-derived mean {mean_sat:.2f}) "
            f"but very low spatial variation (saturation std {sat_std:.3f} < "
            f"{thresholds['sat_std_min']:.3f}, intensity std {pixel_std:.3f} < "
            f"{thresholds['pixel_std_min']:.3f}). This pattern is characteristic "
            "of a flat solid-colour graphic rather than H&E-stained histopathology "
            "tissue. Upload a genuine histopathology slide."
        )

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Image domain screening  (OOD pre-check, runs before model inference)
# ─────────────────────────────────────────────────────────────────────────────

def screen_image_domain(
    image_array: np.ndarray, cancer_type: str
) -> tuple[bool, str, str]:
    """
    Three-signal domain check applied after preprocessing and before model inference.
    No separate classifier model is required.

    All three signals exploit physical properties of medical imaging modalities
    rather than protocol-dependent measurements, making them robust to scanner
    variation, window/level settings, and JPEG compression.

    Signal 1 — Colour saturation ceiling (per-modality):
        Brain MRI and lung CT are near-grayscale by acquisition physics (mean
        HSV saturation < 0.05 in practice).  Colon H&E histology has
        pink/purple staining (typically 0.10–0.35).  Logos, photographs, and
        colourful graphics vastly exceed both ranges.
        Threshold source: SATURATION_THRESHOLD in config.py.
        Rejection reason: "non_medical_image"

    Signal 2 — Colour saturation floor (colon only):
        The H&E staining process always introduces pink/purple colour, even in
        background tissue tiles.  A near-grayscale image (saturation < 0.05)
        submitted to the colon model is almost certainly a CT or MRI scan.
        Threshold source: SATURATION_FLOOR in config.py.
        Rejection reason: "cross_modality"

    Signal 2b — Flat-colour image check (colon only):
        Real H&E slides have spatial variation in staining density (non-zero
        saturation std) and micro-texture from cells and glands (non-zero pixel
        std).  Solid-colour graphics, logos, and uniform colour fills fail both.
        Both must fail simultaneously (AND logic) to trigger rejection —
        conservative to avoid rejecting uniformly-stained tissue patches.
        Uses RGB-derived saturation proxy (no cv2 call) for independence from
        Signal 1/2's HSV computation.
        Threshold source: COLON_FLAT_IMAGE_THRESHOLDS in config.py.
        Rejection reason: "non_medical_image"

    Signal 3 — Intensity profile (brain and lung models only):
        Saturation signals cannot distinguish brain MRI from lung CT — both are
        near-grayscale.  This signal uses pixel intensity distribution shape:

        Brain model: lung CT has a bimodal histogram (dark parenchyma + bright
          mediastinum/bone) that produces high dark_fraction (> 0.50) AND high
          pixel std (> 0.32).  Brain MRI has a more concentrated distribution
          and fails at least one condition.

        Lung model: brain MRI has a smooth uniform bright central region (the
          brain tissue), while lung CT always has dark parenchyma within the
          centre block, making it heterogeneous (centre_std > 0.08).  A smooth
          bright centre (centre_std < 0.08 AND centre_mean > 0.40) indicates
          brain MRI submitted to the wrong model.

        Both conditions in each check use AND logic (conservative): a single
        feature exceedance is insufficient to trigger rejection, preventing
        false rejections of valid scans with unusual acquisition parameters.
        Threshold source: MODALITY_INTENSITY_PROFILES in config.py.
        Rejection reason: "cross_modality"

    Design note — why edge density (Laplacian variance) was evaluated but NOT used:
        Laplacian variance was previously trialled as a signal to catch H&E
        histopathology sent to brain/lung models.  It was removed because
        HRCT and nodule-dense scans routinely produce variance values that
        overlap with histopathology ranges — CT edge density is heavily
        affected by acquisition protocol, window/level, and JPEG compression,
        so no fixed threshold generalises.  Intensity distribution shape (Signal 3)
        operates at a coarser scale and is more stable across CT protocols.

    Args:
        image_array: preprocessed float32 array of shape (1, H, W, 3) in [0,255]
        cancer_type: lower-case modality key ("brain", "lung", "colon", …)

    Returns:
        (is_suspicious: bool, rejection_reason: str, message: str)
        rejection_reason is "non_medical_image" or "cross_modality" when
        suspicious, and "" when the image passes all checks.
    """
    img = image_array[0].astype(np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(img_hsv[:, :, 1])) / 255.0

    # Signal 1: saturation ceiling — catches non-medical (colourful) images and
    #           H&E histopathology sent to brain/lung models (high saturation).
    sat_max = SATURATION_THRESHOLD.get(cancer_type, 0.50)
    if mean_saturation > sat_max:
        return True, "non_medical_image", (
            f"Image colour saturation ({mean_saturation:.2f}) exceeds the "
            f"expected range for {cancer_type} imaging (threshold: {sat_max:.2f}). "
            "Upload a genuine medical scan."
        )

    # Signal 2: saturation floor — catches near-grayscale images (CT/MRI) sent
    #           to the colon (histopathology) model.
    sat_min = SATURATION_FLOOR.get(cancer_type)
    if sat_min is not None and mean_saturation < sat_min:
        return True, "cross_modality", (
            f"Image colour saturation ({mean_saturation:.2f}) is below the "
            f"expected range for {cancer_type} imaging (minimum: {sat_min:.2f}). "
            "Colon histopathology scans show H&E staining colour. "
            "If you submitted a CT or MRI scan, select the correct cancer type."
        )

    # Signal 2b: flat-colour image check (colon only) — catches solid-colour
    #   non-medical graphics (logos, swatches) within the saturation ceiling/floor
    #   window.  Uses RGB-derived statistics; no cv2 call required.
    if cancer_type == "colon":
        is_flat, flat_msg = _check_colon_flat_image(image_array)
        if is_flat:
            return True, "non_medical_image", flat_msg

    # Signal 3: intensity profile — catches brain MRI ↔ lung CT cross-submission.
    #   Brain MRI and lung CT have similar saturation (both near-grayscale), so
    #   Signals 1 and 2 cannot separate them.  Signal 3 uses the *shape* of the
    #   pixel intensity distribution, which differs between the two modalities.
    features = _compute_intensity_features(image_array)
    is_mismatch, mismatch_msg = _check_modality_compatibility(features, cancer_type)
    if is_mismatch:
        return True, "cross_modality", mismatch_msg

    return False, "", ""


# ─────────────────────────────────────────────────────────────────────────────
# Prediction quality assessment
# ─────────────────────────────────────────────────────────────────────────────

def _assess_prediction_quality(
    raw_probs: np.ndarray, cancer_type: str = ""
) -> tuple[bool, str]:
    """
    Apply four statistical criteria to determine whether the model's output
    distribution is reliable enough to return as a diagnosis.

    Criteria (all four must pass; any failure → ambiguous):
      1. Confidence     : max softmax prob ≥ conf_threshold
      2. Margin         : gap between top-2 probs ≥ margin_threshold
      3. Entropy ceiling: H/H_max ≤ entropy_threshold
      4. Entropy floor  : H/H_max ≥ ENTROPY_FLOOR (0.01, modality-invariant)
                          — fires only when p₁ ≳ 99.9%; catches OOD commitment

    H_max = ln(n_classes) makes all entropy bounds scale-invariant across
    3-class (lung), 4-class (brain), and 8-class (colon) models.

    Per-modality thresholds:
      brain / lung (and default): CONF=0.35, MARGIN=0.07, ENTROPY_CEILING=0.95
      colon                     : CONF=0.20, MARGIN=0.02, ENTROPY_CEILING=0.99
        — Kather 2016 low-cellularity classes (empty, stroma, adipose, debris)
          produce conf 20–30% and norm_H 0.96–0.98, which is still well above
          the 8-class random baseline (12.5%).  Global thresholds would
          incorrectly reject these valid tissue predictions.

    Returns:
        (is_ambiguous: bool, reason: str)
    """
    if cancer_type == "colon":
        conf_thr    = COLON_QUALITY_THRESHOLDS["conf_threshold"]
        margin_thr  = COLON_QUALITY_THRESHOLDS["margin_threshold"]
        entropy_thr = COLON_QUALITY_THRESHOLDS["entropy_threshold"]
    else:
        conf_thr    = CONF_THRESHOLD
        margin_thr  = MARGIN_THRESHOLD
        entropy_thr = ENTROPY_THRESHOLD

    confidence = float(np.max(raw_probs))

    sorted_probs = np.sort(raw_probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else 1.0

    n = len(raw_probs)
    raw_entropy = float(-np.sum(raw_probs * np.log(raw_probs + 1e-10)))
    h_max = float(np.log(n)) if n > 1 else 1.0
    norm_entropy = raw_entropy / h_max

    failures = []
    if confidence < conf_thr:
        failures.append(f"confidence {confidence:.0%} < {conf_thr:.0%}")
    if margin < margin_thr:
        failures.append(f"margin {margin:.2f} < {margin_thr:.2f}")
    if norm_entropy > entropy_thr:
        failures.append(f"entropy {norm_entropy:.2f} > {entropy_thr:.2f} (normalised)")
    if norm_entropy < ENTROPY_FLOOR:
        failures.append(
            f"entropy {norm_entropy:.2f} < {ENTROPY_FLOOR:.2f} (normalised) "
            "— extreme over-concentration suggests non-medical input"
        )

    if failures:
        return True, "Ambiguous prediction — " + ", ".join(failures)
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Cancer classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_cancer(image_array: np.ndarray, cancer_type: str) -> dict:
    """
    Run the appropriate cancer classification model, then apply the three-criterion
    quality gate.

    Returns a dict with:
        predicted_class    : human-readable label, or "Invalid Scan" if ambiguous
    """
    if cancer_type not in CLASS_LABELS:
        raise ValueError(
            f"Unknown cancer_type '{cancer_type}'. "
            f"Valid options: {sorted(CLASS_LABELS.keys())}"
        )

    model = _get_model(cancer_type)
    if model is None:
        raise RuntimeError(
            f"Model for '{cancer_type}' is not loaded. "
            "Check /health for missing models."
        )

    raw_probs   = model.predict(image_array, verbose=0)[0]
    class_names = CLASS_LABELS[cancer_type]

    is_ambiguous, reason = _assess_prediction_quality(raw_probs, cancer_type=cancer_type)

    if is_ambiguous:
        logger.info("[classify_cancer] Ambiguous — %s", reason)
        return {"predicted_class": "Invalid Scan"}

    predicted_idx = int(np.argmax(raw_probs))
    raw_label     = class_names[predicted_idx]

    if raw_label in COLON_EXCLUDED_CLASSES:
        logger.info("[classify_cancer] Suppressed class '%s' → Invalid Scan", raw_label)
        return {"predicted_class": "Invalid Scan"}

    predicted_class = LABEL_MAPPING.get(raw_label, raw_label)
    return {"predicted_class": predicted_class}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes, cancer_type: str) -> dict:
    """
    Three-stage inference pipeline.

    Stage 1: Input validation   (file format / size / corruption)
    Stage 2: Domain screening   (colour-saturation OOD pre-check, no extra model)
    Stage 3: Cancer classification + quality gate (four-criterion softmax gate)

    Always returns a dict that maps 1:1 to PredictionResponse schema.
    Never raises — all exceptions are caught and returned as structured errors.
    """
    logger.info("[predict] Request — cancer_type=%r  bytes=%d", cancer_type, len(image_bytes))

    # ── Stage 1: Input validation ────────────────────────────────────────────
    try:
        validate_input_bytes(image_bytes)
    except ValueError as exc:
        logger.warning("[predict] Stage 1 REJECTED — %s", exc)
        return {
            "status" : "rejected",
            "reason" : "invalid_input",
            "message": str(exc),
        }

    # ── Stage 2: Preprocess → domain screening ───────────────────────────────
    clean_type = cancer_type.lower().strip()

    # Lung CT: apply CLAHE to match the training pipeline (train_lung.py).
    # Brain / colon: no CLAHE — those models were not trained with it.
    apply_clahe = clean_type == "lung"
    image_array = preprocess_image(image_bytes, apply_clahe=apply_clahe)

    # Colour-saturation OOD check: reject clearly non-medical images before
    # running the (expensive) cancer model.  Brain MRI and CT are near-grayscale;
    # logos and photographs are far more saturated.
    is_domain_mismatch, domain_reason_code, domain_reason_msg = screen_image_domain(
        image_array, clean_type
    )
    if is_domain_mismatch:
        logger.warning("[predict] Stage 2 DOMAIN MISMATCH (%s) — %s", domain_reason_code, domain_reason_msg)
        return {
            "status" : "rejected",
            "reason" : domain_reason_code,
            "message": domain_reason_msg,
        }

    # ── Stage 3: Classify + quality gate ────────────────────────────────────
    logger.info("[predict] Stage 3 — classifying '%s'", clean_type)
    try:
        result = classify_cancer(image_array, clean_type)
    except ValueError as exc:
        logger.warning("[predict] Invalid cancer type: %s", exc)
        return {
            "status" : "rejected",
            "reason" : "invalid_cancer_type",
            "message": str(exc),
        }
    except RuntimeError as exc:
        logger.error("[predict] Model unavailable: %s", exc)
        return {
            "status" : "error",
            "reason" : "model_unavailable",
            "message": str(exc),
        }

    logger.info("[predict] Done — class=%r", result["predicted_class"])
    return {
        "status"     : "success",
        "cancer_type": cancer_type,
        **result,
    }
