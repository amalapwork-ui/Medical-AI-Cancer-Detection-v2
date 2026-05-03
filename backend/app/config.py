"""
Central configuration for the Medical AI backend.
All file paths are resolved relative to this file so the server can be
started from any working directory (not just backend/).
"""

from pathlib import Path

# ── Directory layout ────────────────────────────────────────────────────────
APP_DIR     = Path(__file__).resolve().parent   # backend/app/
BACKEND_DIR = APP_DIR.parent                    # backend/
MODELS_DIR  = BACKEND_DIR / "models"           # backend/models/

# ── Model file paths ─────────────────────────────────────────────────────────
MODEL_PATHS: dict[str, Path] = {
    "brain" : MODELS_DIR / "brain_cancer_effnet.keras",
    "lung"  : MODELS_DIR / "lung_cancer_final.keras",
    "colon" : MODELS_DIR / "colon_cancer_final.keras",
}

# ── Class labels (must match training-time class_names order) ────────────────
CLASS_LABELS: dict[str, list[str]] = {
    "brain" : ["glioma", "meningioma", "notumor", "pituitary"],
    # IQ-OTH/NCCD dataset — real CT scan classification (3 classes, alphabetical)
    "lung"  : ["benign", "malignant", "normal"],
    # Kather 2016 — 8 tissue classes (alphabetical = Keras inferred order)
    "colon" : ["adipose", "complex", "debris", "empty", "lympho", "mucosa", "stroma", "tumor"],
}

# ── Human-readable display names ─────────────────────────────────────────────
LABEL_MAPPING: dict[str, str] = {
    # Brain
    "glioma"    : "Glioma",
    "meningioma": "Meningioma",
    "pituitary" : "Pituitary Tumor",
    "notumor"   : "No Tumor",
    # Lung — IQ-OTH/NCCD
    "benign"    : "Benign Lung Lesion",
    "malignant" : "Malignant Lung Cancer",
    "normal"    : "Normal Lung",
    # Colon — Kather 2016 (8-class tissue classification)
    "adipose"   : "Adipose Tissue",
    "complex"   : "Complex Glandular Epithelium",
    "debris"    : "Cellular Debris",
    "empty"     : "Background / Empty",
    "lympho"    : "Lymphocytic Infiltrate",
    "mucosa"    : "Normal Mucosa",
    "stroma"    : "Cancer-Associated Stroma",
    "tumor"     : "Colorectal Adenocarcinoma",
}

# ── Confidence-based prediction quality thresholds ───────────────────────────
# These three criteria together determine whether a prediction is reliable
# enough to return to the user. All three must pass; if any fails the response
# carries predicted_class="Invalid Scan" with prediction_status="ambiguous".
#
# CONF_THRESHOLD   — minimum top-class softmax probability.
#                    Set to 0.35: a realistic floor that filters near-random
#                    outputs while accepting valid low-confidence predictions
#                    (e.g. a lung CT returning 53%/27%/20% should not be
#                    rejected — EfficientNetV2S with focal loss typically
#                    produces 50–65% top-class probs on real-world CT images).
#
# MARGIN_THRESHOLD — minimum gap between the top-2 softmax probabilities.
#                    0.07 = the model must show a clear preference for one class.
#
#                    Calibrated to the "notumor" (no-tumour) brain class:
#                    EfficientNetV2S trained on the masoudnickparvar dataset
#                    produces smaller margins on notumor (typically 7–12%) than
#                    on positive-tumour classes (typically 15–40%), because:
#                      (a) the dataset has ~395 notumor images vs ~826 for each
#                          tumour class — roughly 2× less training data.
#                      (b) detecting the absence of a feature is inherently harder
#                          than detecting its presence; the model is less decisive.
#                    0.10 incorrectly rejected valid 7–9% margin notumor
#                    predictions.  0.07 still filters true near-ties (margin < 7%)
#                    while allowing small-but-decisive notumor margins through.
#
# ENTROPY_THRESHOLD— maximum *normalised* Shannon entropy H/H_max.
#                    H_max = ln(n_classes); normalising makes this threshold
#                    consistent across 3-class (lung), 4-class (brain), and
#                    8-class (colon) models. 0.95 = near-uniform distributions
#                    are rejected while moderate uncertainty is accepted.
CONF_THRESHOLD   : float = 0.35
MARGIN_THRESHOLD : float = 0.07
ENTROPY_THRESHOLD: float = 0.95   # normalised H/H_max; compare after dividing

# ── OOD detection: softmax distribution floor ─────────────────────────────────
# Rejects predictions whose softmax distribution is unnaturally concentrated —
# a hallmark of OOD forced-commitment where the backbone strongly activates one
# head despite no valid class being present, rather than genuine learned certainty.
#
# CALIBRATION NOTE — why 0.01, not 0.05:
#   For the 4-class brain model, H_max = ln(4) ≈ 1.386.
#   ENTROPY_FLOOR = 0.05 corresponds to H < 0.0693, which fires at p₁ ≈ 99% —
#   well within the range of legitimate high-confidence predictions from a trained
#   EfficientNetV2S model on clear medical images (pituitary adenomas and clear
#   no-tumor brain scans routinely score 99–99.7%).  0.05 was therefore
#   miscalibrated: it rejected valid predictions as OOD.
#
#   ENTROPY_FLOOR = 0.01 corresponds to H < 0.01 × H_max, which fires at
#   p₁ ≈ 99.9% for 4-class models — a regime that no trained medical model
#   reaches on valid in-distribution images.  By that point essentially all
#   probability mass is on one class, which is the hallmark of OOD commitment.
#
#   Equivalent thresholds per model:
#     brain  (4 classes, H_max=1.386): fires when p₁ ≳ 99.9%
#     lung   (3 classes, H_max=1.099): fires when p₁ ≳ 99.9%
#     colon  (8 classes, H_max=2.079): fires when p₁ ≳ 99.8%
ENTROPY_FLOOR: float = 0.01

# ── OOD detection: per-modality colour saturation gate ────────────────────────
# Brain MRI and lung CT are near-grayscale; their mean HSV saturation is
# typically < 0.05. Colon H&E histology has pink/purple staining (typically
# 0.10–0.35). Images with mean saturation above the per-modality threshold are
# screened out before model inference — no separate classifier model required.
#
# These thresholds are deliberately permissive to minimise false rejections:
#   brain / lung : MRI and CT are essentially grayscale; 0.15 has wide margin
#   colon        : Real H&E histopathology reaches mean saturation 0.55–0.70.
#                  Eosin (pink) and haematoxylin (blue-purple) are strong dyes;
#                  tumour glands, stroma, and lymphocytic infiltrates routinely
#                  produce mean HSV saturation in this range.  0.85 is a hard
#                  ceiling above any clinically observed H&E value — anything
#                  above it is a neon graphic or heavily edited image, not tissue.
#                  Flat solid-colour graphics within 0.35–0.85 (logos, swatches)
#                  are caught separately by Signal 2b (COLON_FLAT_IMAGE_THRESHOLDS).
SATURATION_THRESHOLD: dict[str, float] = {
    "brain": 0.15,
    "lung" : 0.15,
    "colon": 0.85,
}

# ── Cross-modality detection thresholds ───────────────────────────────────────
# Used by screen_image_domain() to detect medical images submitted to the
# wrong cancer-type model.  No new trained model is required — these signals
# exploit per-modality image statistics.

# SATURATION_FLOOR — minimum expected mean HSV saturation per modality.
#   Colon H&E histopathology always contains some pink/purple staining from the
#   haematoxylin and eosin process; mean saturation is typically 0.10–0.35.
#   A near-grayscale image (saturation < 0.05) sent to the colon model is
#   almost certainly a CT or MRI scan.  CT/MRI images typically have mean
#   saturation < 0.02; the 0.05 floor leaves ample margin for lightly stained
#   histopathology slides (including Kather 2016 "empty" background tiles,
#   which still carry residual staining and typically measure > 0.06).
SATURATION_FLOOR: dict[str, float] = {
    "colon": 0.05,
}


# ── Colon flat-image check (Signal 2b) ────────────────────────────────────────
# Detects solid-colour non-medical images (logos, colour swatches, uniform fills)
# that fall within the colon saturation ceiling (0.05–0.85) and would otherwise
# pass Signals 1 and 2.
#
# Real H&E slides have:
#   • Spatial variation in staining density  → sat_std (RGB-derived) > 0
#   • Micro-texture from cells, gland walls, and nuclei → pixel_std > 0
#
# Solid-colour graphics fail both measures.  Both must be below their thresholds
# (AND logic) to trigger rejection — conservative to avoid rejecting uniformly-
# stained tissue patches (e.g. Kather 2016 "empty" background tiles, mucosa).
#
# Fields:
#   suspicious_sat_min — only run the texture check when RGB-derived mean
#                        saturation exceeds this value.  Below it the image is
#                        lightly coloured; Signal 2 and the quality gate suffice.
#   sat_std_min        — minimum std of RGB-derived per-pixel saturation proxy.
#                        < 0.04 in a solid uniform-colour image.
#   pixel_std_min      — minimum std of grayscale pixel intensity.
#                        < 0.04 in a solid uniform-colour image.
COLON_FLAT_IMAGE_THRESHOLDS: dict[str, float] = {
    "suspicious_sat_min": 0.35,
    "sat_std_min"       : 0.04,
    "pixel_std_min"     : 0.04,
}


# ── Colon-specific quality gate thresholds ────────────────────────────────────
# The global quality gate (CONF_THRESHOLD=0.35, MARGIN_THRESHOLD=0.07,
# ENTROPY_THRESHOLD=0.95) is calibrated for brain MRI (4 classes, random
# baseline 25%) and lung CT (3 classes, random baseline 33%).  The Kather 2016
# colon model has 8 tissue classes (random baseline 12.5%, H_max = ln(8) ≈ 2.079).
#
# Low-cellularity tissue classes — empty background tiles, sparse stroma,
# adipose tissue, and cellular debris — produce weaker softmax signals than
# tumour classes because they share low-information features.  For these valid
# inputs the model may return:
#     conf ≈ 20–30%  (above random 12.5% but below the global floor of 35%)
#     margin ≈ 2–6%  (small but decisive for an 8-class model)
#     norm_H ≈ 0.96–0.98  (high from multi-class ambiguity; exceeds global 0.95)
#
# Rationale for each colon-specific value:
#   conf_threshold   = 0.20:  1.6× above random (12.5%); filters truly random
#                             outputs while accepting uncertain tissue classes.
#   margin_threshold = 0.02:  minimum decisive preference in an 8-class model;
#                             smaller than global 0.07 because 8-class distributions
#                             naturally produce smaller margins than 3- or 4-class.
#   entropy_threshold= 0.99:  accepts high-entropy outputs from uncertain-but-valid
#                             low-cellularity patches; only rejects truly flat
#                             distributions (norm_H > 0.99).
#
# ENTROPY_FLOOR (0.01) is NOT overridden here — OOD over-commitment detection
# applies equally to all modalities.  For 8 classes, the floor fires only when
# p₁ ≳ 99.8%, a regime no trained EfficientNetB2 reaches on valid tissue.
COLON_QUALITY_THRESHOLDS: dict[str, float] = {
    "conf_threshold"   : 0.20,
    "margin_threshold" : 0.02,
    "entropy_threshold": 0.99,
}


# ── Cross-modality: intensity profile gate (brain ↔ lung) ────────────────────
# Saturation signals cannot distinguish brain MRI from lung CT — both are
# near-grayscale by physics.  These supplementary intensity checks exploit
# the different *distribution shape* of each modality's pixel values:
#
#   Lung CT  : bimodal histogram — very dark lung parenchyma (~0–50/255) plus
#              bright mediastinum/bone (~150–220/255).  This produces a high
#              dark-pixel fraction AND high pixel standard deviation.
#
#   Brain MRI: smoother, more concentrated histogram — brain tissue occupies
#              the mid-range (~80–200/255); dark pixels are limited to the
#              peripheral background outside the skull.  The central image
#              region is smooth (low within-centre std) and bright (elevated
#              centre mean).
#
# Each modality has TWO conditions that must BOTH be exceeded (AND logic).
# This is intentionally conservative: a single feature exceedance is
# insufficient to trigger rejection, preventing false rejections of valid scans
# with unusual acquisition parameters.  Cross-modality submissions that pass
# this check are still handled downstream by the softmax quality gate.
#
# Threshold calibration note: these values are based on the intensity
# statistics described above.  ct_dark_fraction_min = 0.50 catches the typical
# lung CT (≥50% of pixels are very dark — lung parenchyma + circular FOV
# background).  brain_centre_std_max = 0.08 catches smooth uniform brain tissue
# in the central block while allowing for the small variation always present in
# real CT parenchyma (even mostly-mediastinum centre blocks have centre_std
# ≈ 0.14 due to vessel/airway variation).
MODALITY_INTENSITY_PROFILES: dict[str, dict[str, float]] = {
    "brain": {
        # Reject if image shows a CT-bimodal intensity signature.
        # Brain MRI background is ~20–40% of the image; lung CT parenchyma
        # typically pushes dark fraction above 50% while simultaneously
        # producing high pixel variance (bimodal distribution).
        "ct_dark_fraction_min": 0.50,   # > 50% pixels below 0.20 normalised
        "ct_std_min"          : 0.32,   # pixel std > 0.32 → bimodal spread
    },
    "lung": {
        # Reject if the image shows the brain-MRI pattern: a smooth, elevated
        # centre region (uniform grey/white matter) with a meaningful dark
        # peripheral background (the skull exterior / image background).
        #
        # Three conditions, ALL must hold (AND logic):
        #   centre_std  < 0.08  — centre is smooth; no dark parenchyma present
        #   centre_mean > 0.40  — centre is bright; brain tissue level
        #   dark_fraction > 0.15 — some peripheral dark area exists (background
        #                          outside the skull).  This third condition
        #                          prevents false rejection of completely uniform
        #                          images (test fixtures, flat screenshots) which
        #                          have dark_fraction ≈ 0 and are not brain MRI.
        "brain_centre_std_max" : 0.08,   # smooth centre  → no lung parenchyma
        "brain_centre_mean_min": 0.40,   # elevated mean  → brain tissue present
        "brain_dark_fraction_min": 0.15, # some background → real MRI, not blank
    },
}

# Edge density (Laplacian variance) was evaluated as a third signal but was
# REMOVED because it produces false rejections of valid scans in practice:
#
#   High-resolution lung CT (HRCT), nodule-dense CT, and images exported with
#   a narrow JPEG window can easily produce Laplacian variance > 1 500, which
#   any fixed threshold would mis-classify as histopathology.  Unlike colour
#   saturation — which exploits a physical property of imaging physics (MRI/CT
#   are grayscale by necessity; H&E staining always introduces colour) — edge
#   density is heavily affected by CT protocol, window/level settings, and
#   JPEG compression artefacts.  No single threshold generalises reliably.
#
#   Cross-modality cases that are not caught by the two saturation signals are
#   handled downstream by the softmax quality gate (_assess_prediction_quality):
#   the lung model applied to a histopathology slide produces either a near-
#   uniform distribution (fails entropy ceiling / margin) or an anomalously
#   peaked distribution (fails entropy floor), both of which return "ambiguous".

# ── Input validation ─────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024   # 20 MB
MIN_FILE_SIZE_BYTES: int = 1_000              # 1 KB (catches truncated uploads)
IMAGE_SIZE: tuple[int, int] = (224, 224)

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
})
