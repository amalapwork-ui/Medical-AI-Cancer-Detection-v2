"""
Inference pipeline test suite.

Tests cover:
  - Input validation (corrupted, too small, oversized, wrong format)
  - Preprocessing consistency
  - Prediction quality assessment (confidence / margin / entropy gate)
  - Cancer classification (mock)
  - Full pipeline integration with mocked models
  - Edge cases: solid colour images, noise, very small dimensions

Run with:
    cd project_root
    pytest tests/test_inference.py -v
"""

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ── Path setup: allow imports from backend/app ───────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub tensorflow so tests run without a GPU / full TF install.
# keras.Model must be present because predict.py has a module-level annotation
# `_models: dict[str, tf.keras.Model] = {}` which Python evaluates at import time.
tf_stub = types.ModuleType("tensorflow")
tf_stub.keras = types.ModuleType("tensorflow.keras")
tf_stub.keras.Model = MagicMock()
tf_stub.keras.models = types.ModuleType("tensorflow.keras.models")
tf_stub.keras.models.load_model = MagicMock()
sys.modules.setdefault("tensorflow", tf_stub)
sys.modules.setdefault("tensorflow.keras", tf_stub.keras)
sys.modules.setdefault("tensorflow.keras.models", tf_stub.keras.models)

# Stub cv2 so predict.py imports cleanly without opencv installed.
# Default HSV: saturation uint8=20 → mean ≈ 0.08.  Passes all per-modality
# saturation checks without touching any threshold: below brain/lung ceiling
# (0.15), above colon floor (0.05), and below colon ceiling (0.50).
# Laplacian stub is kept for completeness but is not called by production code
# (edge density was removed as a signal — see config.py for rationale).
cv2_stub = types.ModuleType("cv2")
_clahe_stub = MagicMock()
_clahe_stub.apply = lambda x: x
_default_hsv = np.zeros((224, 224, 3), dtype=np.uint8)
_default_hsv[:, :, 1] = 20
cv2_stub.createCLAHE    = MagicMock(return_value=_clahe_stub)
cv2_stub.cvtColor       = MagicMock(return_value=_default_hsv)
cv2_stub.COLOR_RGB2HSV  = 40
sys.modules.setdefault("cv2", cv2_stub)

from backend.app import predict as predict_module
from backend.app.predict import (
    _assess_prediction_quality,
    _check_colon_flat_image,
    _check_modality_compatibility,
    _compute_intensity_features,
    classify_cancer,
    preprocess_image,
    screen_image_domain,
    validate_input_bytes,
)
from backend.app.config import CLASS_LABELS, IMAGE_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_bytes(
    width: int = 300,
    height: int = 300,
    mode: str = "RGB",
    color=(128, 128, 128),
    fmt: str = "JPEG",
) -> bytes:
    """Create a minimal in-memory image and return its bytes."""
    img = Image.new(mode, (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_mock_model(output_probs: list[float]) -> MagicMock:
    """Return a mock Keras model whose .predict() returns the given probabilities."""
    model    = MagicMock()
    np_probs = np.array([output_probs], dtype=np.float32)
    model.predict = MagicMock(return_value=np_probs)
    return model


def _make_ct_like_bytes() -> bytes:
    """
    Create a JPEG with a lung-CT-like intensity profile.
    ~67% very dark pixels (lung parenchyma) + ~33% very bright (bone/contrast).

    After preprocess_image the decoded float32 array has:
        dark_fraction ≈ 0.65+ > 0.50 (ct_dark_fraction_min)
        pixel std      ≈ 0.44+ > 0.32 (ct_std_min)
    → Signal 3 rejects this image when submitted to the brain model.

    Pixel values chosen with headroom for JPEG quality-95 quantisation:
    dark region uses value=3 (threshold is 51), bright uses value=240.
    """
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:150, :, :] = 3      # ~67% very dark  (lung parenchyma + FOV background)
    arr[150:, :, :] = 240    # ~33% very bright (bone / contrast agents)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_mri_like_bytes() -> bytes:
    """
    Create a JPEG with a brain-MRI-like intensity profile.
    Smooth uniform bright centre (brain tissue) inside a dark peripheral border.

    After preprocess_image the decoded float32 array has:
        dark_fraction ≈ 0.32 < 0.50 (passes brain Signal 3 check)
        pixel std      ≈ 0.22 < 0.32 (passes brain Signal 3 check)
        centre_std     < 0.08        (brain_centre_std_max)
        centre_mean    > 0.40        (brain_centre_mean_min)
    → Signal 3 rejects this image when submitted to the lung model.

    JPEG quality=95 preserves the uniform interior well; DCT artefacts in a
    uniform 130-valued block are typically ±2-4 counts, giving centre_std
    well below 0.08 in normalised [0,1] space.
    """
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:20, :, :] = 10           # top background (dark)
    arr[204:, :, :] = 10          # bottom background (dark)
    arr[20:204, :20, :] = 10      # left background (dark)
    arr[20:204, 204:, :] = 10     # right background (dark)
    arr[20:204, 20:204, :] = 130  # brain tissue (smooth, medium-bright)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_he_like_bytes() -> bytes:
    """
    Create a JPEG that simulates H&E histopathology.

    Two-region layout (eosin pink + haematoxylin purple) with pixel noise to
    simulate cell and gland micro-texture.

    After preprocess_image the decoded float32 array has:
        RGB-derived mean saturation ≈ 0.68 > 0.35  (suspicious_sat_min)
        sat_std   ≈ 0.07  > 0.04  (sat_std_min)   → passes Signal 2b
        pixel_std ≈ 0.09  > 0.04  (pixel_std_min) → passes Signal 2b

    Colours chosen with headroom for JPEG quality-95 quantisation.
    """
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:112, :, :] = [210, 80, 100]    # eosin-pink region
    arr[112:, :, :] = [60, 40, 160]     # haematoxylin-purple region
    rng   = np.random.default_rng(42)
    noise = rng.integers(-10, 10, arr.shape, dtype=np.int16)
    arr   = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img   = Image.fromarray(arr, mode="RGB")
    buf   = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_flat_color_bytes(color: tuple = (200, 80, 100)) -> bytes:
    """
    Create a JPEG solid-colour image (simulates a logo / colour swatch).

    After preprocess_image the decoded float32 array has:
        RGB-derived mean saturation ≈ 0.60 > 0.35  (suspicious_sat_min)
        sat_std   ≈ 0.005 < 0.04  (sat_std_min)   → fails Signal 2b
        pixel_std ≈ 0.005 < 0.04  (pixel_std_min) → fails Signal 2b
        → both conditions fail → rejected by Signal 2b

    JPEG quality=95 keeps quantisation artefacts well below the 0.04 threshold.
    """
    img = Image.new("RGB", (224, 224), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateInputBytes:

    def test_valid_jpeg_passes(self):
        data = _make_image_bytes()
        validate_input_bytes(data)  # must not raise

    def test_valid_png_passes(self):
        # Solid-colour PNGs compress to <1 000 bytes and fail the size check.
        # Use a noise image: high entropy prevents deflate from compressing it small.
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        validate_input_bytes(buf.getvalue())  # must not raise

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError, match="too small"):
            validate_input_bytes(b"")

    def test_tiny_bytes_raises(self):
        with pytest.raises(ValueError, match="too small"):
            validate_input_bytes(b"x" * 100)

    def test_oversized_raises(self):
        oversized = b"x" * (21 * 1024 * 1024)   # 21 MB
        with pytest.raises(ValueError, match="limit"):
            validate_input_bytes(oversized)

    def test_corrupted_bytes_raises(self):
        with pytest.raises(ValueError):
            validate_input_bytes(b"not-an-image" * 500)

    def test_truncated_jpeg_raises(self):
        valid   = _make_image_bytes()
        corrupt = valid[: int(len(valid) * 0.6)]
        with pytest.raises(ValueError):
            validate_input_bytes(corrupt)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessImage:

    def test_output_shape(self):
        data = _make_image_bytes(width=512, height=512)
        arr  = preprocess_image(data)
        assert arr.shape == (1, *IMAGE_SIZE, 3), f"Expected (1,224,224,3), got {arr.shape}"

    def test_output_dtype_float32(self):
        data = _make_image_bytes()
        arr  = preprocess_image(data)
        assert arr.dtype == np.float32

    def test_values_in_0_255_range(self):
        data = _make_image_bytes(color=(200, 100, 50))
        arr  = preprocess_image(data)
        assert arr.min() >= 0.0
        assert arr.max() <= 255.0

    def test_grayscale_converted_to_rgb(self):
        img = Image.new("L", (256, 256), 128)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        arr = preprocess_image(buf.getvalue())
        assert arr.shape[-1] == 3, "Grayscale must be converted to 3-channel RGB"

    def test_rgba_converted_to_rgb(self):
        img = Image.new("RGBA", (256, 256), (10, 20, 30, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        arr = preprocess_image(buf.getvalue())
        assert arr.shape[-1] == 3

    def test_tiny_image_upscaled(self):
        data = _make_image_bytes(width=32, height=32)
        arr  = preprocess_image(data)
        assert arr.shape == (1, 224, 224, 3)

    def test_large_image_downscaled(self):
        data = _make_image_bytes(width=1024, height=1024)
        arr  = preprocess_image(data)
        assert arr.shape == (1, 224, 224, 3)

    def test_non_square_image(self):
        data = _make_image_bytes(width=640, height=480)
        arr  = preprocess_image(data)
        assert arr.shape == (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Image domain screening
# ─────────────────────────────────────────────────────────────────────────────

class TestImageDomainScreen:
    """Tests for screen_image_domain() — two-signal saturation domain check.

    screen_image_domain() returns (is_suspicious: bool, reason_code: str, message: str).

    Both signals use colour saturation, which is determined by imaging physics
    and is therefore robust to CT protocol variation, window/level settings, and
    compression artefacts (unlike edge density, which was removed for causing
    false rejections of valid high-contrast CT scans).

    cv2 is patched in each test so tests run without opencv.
    Saturation channel is uint8 (0–255); threshold comparisons use /255 normalisation.
    """

    def _arr(self) -> np.ndarray:
        return preprocess_image(_make_image_bytes())

    def _hsv(self, sat_uint8: int) -> np.ndarray:
        """Return (224,224,3) uint8 HSV array with uniform saturation channel."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 1] = sat_uint8
        return arr

    def _patch_cv2(self, m, sat_uint8: int):
        """Configure the cv2 mock with controlled saturation."""
        m.cvtColor.return_value = self._hsv(sat_uint8)
        m.COLOR_RGB2HSV = 40

    # ── Brain: Signal 1 (saturation ceiling 0.15) ─────────────────────────────

    def test_near_grayscale_brain_passes(self):
        # saturation 10/255 ≈ 0.04 — typical MRI, well below 0.15 ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=10)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "brain")
        assert is_suspicious is False

    def test_colorful_logo_brain_rejected(self):
        # saturation 100/255 ≈ 0.39 — far above 0.15 brain ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=100)
            is_suspicious, reason_code, message = screen_image_domain(self._arr(), "brain")
        assert is_suspicious is True
        assert reason_code == "non_medical_image"
        assert "saturation" in message

    def test_just_below_brain_threshold_passes(self):
        # saturation 37/255 ≈ 0.145 — just under 0.15 ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=37)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "brain")
        assert is_suspicious is False

    def test_just_above_brain_threshold_rejected(self):
        # saturation 39/255 ≈ 0.153 — just over 0.15 ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=39)
            is_suspicious, reason_code, _ = screen_image_domain(self._arr(), "brain")
        assert is_suspicious is True
        assert reason_code == "non_medical_image"

    def test_high_contrast_ct_saturation_still_passes_brain(self):
        # Even a high-contrast CT (saturation ~0.10 after JPEG compression)
        # passes if sent to brain — Signal 1 ceiling is 0.15.
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=25)   # ≈ 0.10
            is_suspicious, _, _ = screen_image_domain(self._arr(), "brain")
        assert is_suspicious is False

    # ── Lung: Signal 1 (saturation ceiling 0.15) ─────────────────────────────

    def test_near_grayscale_lung_passes(self):
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=10)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "lung")
        assert is_suspicious is False

    def test_colorful_image_lung_rejected(self):
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=100)
            is_suspicious, reason_code, _ = screen_image_domain(self._arr(), "lung")
        assert is_suspicious is True
        assert reason_code == "non_medical_image"

    def test_high_contrast_hrct_saturation_passes_lung(self):
        # HRCT scans with complex bronchial detail can have saturation ~0.08.
        # The 0.15 ceiling gives headroom for real-world variation.
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=20)   # ≈ 0.08
            is_suspicious, _, _ = screen_image_domain(self._arr(), "lung")
        assert is_suspicious is False

    # ── Colon: Signal 1 (saturation ceiling 0.50) ────────────────────────────

    def test_he_staining_colon_passes(self):
        # saturation 100/255 ≈ 0.39 — typical H&E pink/purple; 0.05 floor < 0.39 < 0.50 ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=100)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is False

    def test_oversaturated_image_colon_rejected(self):
        # saturation 230/255 ≈ 0.90 — above new 0.85 colon hard ceiling.
        # (The old ceiling was 0.50 which incorrectly rejected valid H&E at 0.55–0.65.)
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=230)
            is_suspicious, reason_code, message = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is True
        assert reason_code == "non_medical_image"
        assert "saturation" in message

    def test_valid_histopathology_moderate_high_saturation_passes(self):
        # saturation 140/255 ≈ 0.55 — previously rejected by 0.50 ceiling;
        # must now pass the raised 0.85 ceiling.  The default test fixture is a
        # solid grey image, so Signal 2b (flat-colour check) does not trigger
        # (RGB-derived mean saturation ≈ 0 < 0.35 suspicious_sat_min).
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=140)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is False

    def test_valid_histopathology_high_saturation_passes(self):
        # saturation 180/255 ≈ 0.71 — typical strongly-stained tumour or stroma.
        # Must pass the 0.85 ceiling.
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=180)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is False

    def test_just_below_new_colon_ceiling_passes(self):
        # saturation 215/255 ≈ 0.843 — just below 0.85 ceiling.
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=215)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is False

    def test_just_above_new_colon_ceiling_rejected(self):
        # saturation 220/255 ≈ 0.863 — just above 0.85 ceiling.
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=220)
            is_suspicious, reason_code, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is True
        assert reason_code == "non_medical_image"

    # ── Colon: Signal 2 (saturation floor 0.05) ──────────────────────────────

    def test_ct_mri_sent_to_colon_rejected_by_saturation_floor(self):
        # Near-grayscale image (CT/MRI): saturation 5/255 ≈ 0.02 < 0.05 floor → cross_modality
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=5)
            is_suspicious, reason_code, message = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is True
        assert reason_code == "cross_modality"
        assert "saturation" in message

    def test_lightly_stained_colon_passes_floor(self):
        # saturation 15/255 ≈ 0.06 — just above the 0.05 floor; far below 0.50 ceiling
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=15)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "colon")
        assert is_suspicious is False

    # ── Unknown modality ──────────────────────────────────────────────────────

    def test_unknown_modality_uses_default_saturation_threshold(self):
        # Unknown cancer_type defaults to 0.50 ceiling; saturation 100/255 ≈ 0.39 → passes
        with patch("backend.app.predict.cv2") as m:
            self._patch_cv2(m, sat_uint8=100)
            is_suspicious, _, _ = screen_image_domain(self._arr(), "kidney")
        assert is_suspicious is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Signal 2b — colon flat-image check
# ─────────────────────────────────────────────────────────────────────────────

class TestColonFlatImageCheck:
    """
    Tests for Signal 2b: _check_colon_flat_image().

    Catches solid-colour non-medical images (logos, swatches) that have enough
    colour to pass the saturation ceiling/floor but lack the spatial variation
    and micro-texture of real H&E slides.

    All tests operate on raw numpy arrays via preprocess_image — no cv2 mock
    required because _check_colon_flat_image uses RGB pixel values directly.
    """

    # ── Unit: _check_colon_flat_image ────────────────────────────────────────

    def test_flat_pink_image_rejected(self):
        # Solid (200, 80, 100) pink image: RGB-derived mean_sat ≈ 0.60 > 0.35,
        # sat_std ≈ 0 < 0.04, pixel_std ≈ 0 < 0.04 → both conditions fail → reject.
        arr      = preprocess_image(_make_flat_color_bytes(color=(200, 80, 100)))
        is_flat, msg = _check_colon_flat_image(arr)
        assert is_flat is True
        assert "saturation" in msg.lower() or "flat" in msg.lower() or "colour" in msg.lower()

    def test_flat_purple_image_rejected(self):
        # Different flat colour — same rejection logic.
        arr = preprocess_image(_make_flat_color_bytes(color=(80, 40, 180)))
        is_flat, _ = _check_colon_flat_image(arr)
        assert is_flat is True

    def test_he_textured_image_passes(self):
        # Two-tone pink/purple image with noise: sat_std > 0.04, pixel_std > 0.04.
        arr = preprocess_image(_make_he_like_bytes())
        is_flat, _ = _check_colon_flat_image(arr)
        assert is_flat is False

    def test_low_saturation_image_skips_check(self):
        # Solid grey (128,128,128): RGB-derived mean_sat = 0 < 0.35 → check skipped.
        arr = preprocess_image(_make_image_bytes(color=(128, 128, 128)))
        is_flat, _ = _check_colon_flat_image(arr)
        assert is_flat is False

    def test_flat_image_message_contains_threshold_values(self):
        # Rejection message must reference the measured values.
        arr = preprocess_image(_make_flat_color_bytes(color=(200, 80, 100)))
        is_flat, msg = _check_colon_flat_image(arr)
        assert is_flat is True
        assert "0.04" in msg    # threshold value appears in message

    def test_textured_colon_returns_empty_message_on_pass(self):
        arr = preprocess_image(_make_he_like_bytes())
        is_flat, msg = _check_colon_flat_image(arr)
        assert is_flat is False
        assert msg == ""

    # ── Integration: Signal 2b through screen_image_domain ───────────────────

    def test_flat_color_colon_pipeline_rejected(self):
        # Solid pink image: passes Signals 1+2 (cv2 mock gives sat=0.08),
        # then Signal 2b fires because real RGB pixels are uniform.
        with patch("backend.app.predict.cv2") as m:
            hsv = np.zeros((224, 224, 3), dtype=np.uint8)
            hsv[:, :, 1] = 20    # saturation 0.078 — passes ceiling and floor
            m.cvtColor.return_value = hsv
            m.COLOR_RGB2HSV = 40
            arr = preprocess_image(_make_flat_color_bytes(color=(200, 80, 100)))
            is_suspicious, reason, msg = screen_image_domain(arr, "colon")
        assert is_suspicious is True
        assert reason == "non_medical_image"
        assert "flat" in msg.lower() or "colour" in msg.lower() or "saturation" in msg.lower()

    def test_he_image_colon_pipeline_passes(self):
        # Textured H&E image: passes Signals 1+2 and Signal 2b.
        with patch("backend.app.predict.cv2") as m:
            hsv = np.zeros((224, 224, 3), dtype=np.uint8)
            hsv[:, :, 1] = 100   # saturation 0.39 — well within colon range
            m.cvtColor.return_value = hsv
            m.COLOR_RGB2HSV = 40
            arr = preprocess_image(_make_he_like_bytes())
            is_suspicious, _, _ = screen_image_domain(arr, "colon")
        assert is_suspicious is False

    def test_signal_2b_not_applied_to_brain_model(self):
        # Flat-colour images submitted to brain model must NOT trigger Signal 2b.
        # (Signal 1 would catch them first via high saturation, but if saturation
        #  is below 0.15 a flat grey image should still pass — the brain model
        #  has no flat-colour check.)
        arr = preprocess_image(_make_flat_color_bytes(color=(200, 80, 100)))
        with patch("backend.app.predict.cv2") as m:
            hsv = np.zeros((224, 224, 3), dtype=np.uint8)
            hsv[:, :, 1] = 10   # saturation 0.039 — below brain ceiling 0.15
            m.cvtColor.return_value = hsv
            m.COLOR_RGB2HSV = 40
            is_suspicious, _, _ = screen_image_domain(arr, "brain")
        assert is_suspicious is False   # brain model has no flat-colour gate


# ─────────────────────────────────────────────────────────────────────────────
# 5. Signal 3 — modality intensity gate
# ─────────────────────────────────────────────────────────────────────────────

class TestModalityIntensityGate:
    """
    Tests for Signal 3: intensity-profile cross-modality detection.

    Signals 1 & 2 use colour saturation — they cannot separate brain MRI from
    lung CT because both are near-grayscale.  Signal 3 uses pixel intensity
    *distribution shape*:

        brain model — rejects CT-bimodal images (dark_fraction > 0.50 AND std > 0.32)
        lung  model — rejects brain-MRI images (centre_std < 0.08 AND centre_mean > 0.40)

    Unit tests operate on raw numpy arrays (exact pixel control).
    Integration tests use the crafted JPEG helpers through the full pipeline.

    Signal 3 uses numpy only; no cv2 calls are made.  Where cv2 is patched
    the mock handles only the saturation computation in Signals 1 & 2.
    """

    # ── Synthetic arrays (exact pixel values) ────────────────────────────────

    @staticmethod
    def _ct_array() -> np.ndarray:
        """Float32 array: ~60% near-black + ~40% very bright → CT-bimodal."""
        arr = np.zeros((1, 224, 224, 3), dtype=np.float32)
        arr[0, :135, :, :] = 5.0     # ~60% very dark  (< 51/255)
        arr[0, 135:, :, :] = 230.0   # ~40% very bright
        return arr

    @staticmethod
    def _mri_array() -> np.ndarray:
        """Float32 array: smooth bright centre + dark border → brain-MRI-like."""
        arr = np.zeros((1, 224, 224, 3), dtype=np.float32)
        arr[0, 20:204, 20:204, :] = 130.0   # brain tissue (uniform, medium-bright)
        # peripheral border remains 0.0 (dark background)
        return arr

    # ── Unit: _compute_intensity_features ────────────────────────────────────

    def test_ct_array_has_high_dark_fraction(self):
        # dark_fraction = 135/224 ≈ 0.603 > 0.50 (ct_dark_fraction_min)
        feats = _compute_intensity_features(self._ct_array())
        assert feats["dark_fraction"] > 0.50, f"dark_fraction={feats['dark_fraction']:.3f}"

    def test_ct_array_has_high_std(self):
        # std ≈ 0.43 (bimodal: 5/255 vs 230/255) > 0.32 (ct_std_min)
        feats = _compute_intensity_features(self._ct_array())
        assert feats["std"] > 0.32, f"std={feats['std']:.3f}"

    def test_mri_array_has_low_centre_std(self):
        # centre [56:168, 56:168] is entirely within the 130-value brain block
        # → centre_std = 0.0 < 0.08 (brain_centre_std_max)
        feats = _compute_intensity_features(self._mri_array())
        assert feats["centre_std"] < 0.08, f"centre_std={feats['centre_std']:.3f}"

    def test_mri_array_has_elevated_centre_mean(self):
        # centre_mean = 130/255 ≈ 0.510 > 0.40 (brain_centre_mean_min)
        feats = _compute_intensity_features(self._mri_array())
        assert feats["centre_mean"] > 0.40, f"centre_mean={feats['centre_mean']:.3f}"

    def test_mri_array_passes_brain_model_dark_fraction_check(self):
        # MRI array dark_fraction ≈ 0.325 < 0.50 → brain Signal 3 must NOT fire.
        feats = _compute_intensity_features(self._mri_array())
        is_mismatch, _ = _check_modality_compatibility(feats, "brain")
        assert is_mismatch is False

    def test_ct_array_triggers_brain_model_rejection(self):
        feats = _compute_intensity_features(self._ct_array())
        is_mismatch, msg = _check_modality_compatibility(feats, "brain")
        assert is_mismatch is True
        assert "cross_modality" not in msg    # message is human-readable, not the code
        assert "CT" in msg or "lung" in msg.lower()

    def test_mri_array_triggers_lung_model_rejection(self):
        feats = _compute_intensity_features(self._mri_array())
        is_mismatch, msg = _check_modality_compatibility(feats, "lung")
        assert is_mismatch is True
        assert "brain" in msg.lower() or "MRI" in msg

    def test_ct_array_passes_lung_model_check(self):
        # CT centre has mixed dark/bright (centre_std ≈ 0.40 > 0.08) → NOT rejected.
        feats = _compute_intensity_features(self._ct_array())
        is_mismatch, _ = _check_modality_compatibility(feats, "lung")
        assert is_mismatch is False

    def test_colon_model_has_no_intensity_profile(self):
        # Colon is fully gated by saturation signals; no intensity check exists.
        feats = _compute_intensity_features(self._ct_array())
        is_mismatch, _ = _check_modality_compatibility(feats, "colon")
        assert is_mismatch is False

    # ── Integration: screen_image_domain with raw arrays ─────────────────────

    def test_ct_array_rejected_by_brain_model_in_screen(self):
        # Signal 3 fires; Signals 1 & 2 pass (saturation mock = 0.08).
        with patch("backend.app.predict.cv2") as m:
            m.cvtColor.return_value = _default_hsv
            m.COLOR_RGB2HSV = 40
            is_suspicious, reason, msg = screen_image_domain(self._ct_array(), "brain")
        assert is_suspicious is True
        assert reason == "cross_modality"
        assert "CT" in msg or "lung" in msg.lower()

    def test_mri_array_rejected_by_lung_model_in_screen(self):
        # Signal 3 fires for the lung model (smooth bright centre).
        with patch("backend.app.predict.cv2") as m:
            m.cvtColor.return_value = _default_hsv
            m.COLOR_RGB2HSV = 40
            is_suspicious, reason, msg = screen_image_domain(self._mri_array(), "lung")
        assert is_suspicious is True
        assert reason == "cross_modality"
        assert "brain" in msg.lower() or "MRI" in msg

    def test_valid_mri_array_passes_brain_model(self):
        with patch("backend.app.predict.cv2") as m:
            m.cvtColor.return_value = _default_hsv
            m.COLOR_RGB2HSV = 40
            is_suspicious, _, _ = screen_image_domain(self._mri_array(), "brain")
        assert is_suspicious is False

    def test_valid_ct_array_passes_lung_model(self):
        with patch("backend.app.predict.cv2") as m:
            m.cvtColor.return_value = _default_hsv
            m.COLOR_RGB2HSV = 40
            is_suspicious, _, _ = screen_image_domain(self._ct_array(), "lung")
        assert is_suspicious is False

    # ── Full pipeline: JPEG images through predict.predict ───────────────────

    def _all_mocks(self):
        return {
            "brain": _make_mock_model([0.90, 0.04, 0.03, 0.03]),
            "lung" : _make_mock_model([0.05, 0.03, 0.92]),
            "colon": _make_mock_model([0.01] * 7 + [0.93]),
        }

    def test_ct_jpeg_rejected_by_brain_pipeline(self):
        # Test 4 (user requirement): CT → brain model → MUST be rejected.
        with patch.dict(predict_module._models, self._all_mocks()):
            result = predict_module.predict(_make_ct_like_bytes(), "brain")
        assert result["status"] == "rejected"
        assert result["reason"] == "cross_modality"

    def test_ct_jpeg_accepted_by_lung_pipeline(self):
        # CT sent to correct model must succeed.
        with patch.dict(predict_module._models, self._all_mocks()):
            result = predict_module.predict(_make_ct_like_bytes(), "lung")
        assert result["status"] == "success"

    def test_mri_jpeg_rejected_by_lung_pipeline(self):
        # Test 5 (user requirement): MRI → lung model → MUST be rejected.
        with patch.dict(predict_module._models, self._all_mocks()):
            result = predict_module.predict(_make_mri_like_bytes(), "lung")
        assert result["status"] == "rejected"
        assert result["reason"] == "cross_modality"

    def test_mri_jpeg_accepted_by_brain_pipeline(self):
        # Brain MRI sent to correct model must succeed.
        with patch.dict(predict_module._models, self._all_mocks()):
            result = predict_module.predict(_make_mri_like_bytes(), "brain")
        assert result["status"] == "success"

    # ── Test 9: same CT image tested across all three models ─────────────────

    def test_ct_only_accepted_by_lung_model(self):
        """
        Test 9: the same CT intensity profile must be accepted only by the
        lung model.  Brain model rejects via Signal 3 (intensity profile).
        Colon model has no intensity profile entry — its CT gate is saturation
        Signal 2 (tested separately); here we verify the intensity-profile
        layer does not interfere with colon.
        """
        ct = self._ct_array()
        feats = _compute_intensity_features(ct)

        brain_mismatch, _ = _check_modality_compatibility(feats, "brain")
        lung_mismatch,  _ = _check_modality_compatibility(feats, "lung")
        colon_mismatch, _ = _check_modality_compatibility(feats, "colon")

        assert brain_mismatch is True,  "CT must be rejected by brain model"
        assert lung_mismatch  is False, "CT must be accepted by lung model"
        assert colon_mismatch is False, "Colon intensity gate must not fire (colon uses saturation)"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prediction quality assessment
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionQuality:
    """Tests for _assess_prediction_quality() — the three-criterion quality gate.

    Thresholds: CONF_THRESHOLD=0.35, MARGIN_THRESHOLD=0.07,
    ENTROPY_THRESHOLD=0.95 (normalised H/H_max).

    Key design intent: valid real-world CT scans returning ~50–65% top-class
    probability must NOT be rejected — EfficientNetV2S with focal loss
    naturally produces these lower confidence values.
    """

    def test_high_confidence_passes(self):
        # conf=0.92 ≥ 0.35, margin=0.88 ≥ 0.10, norm_H very low → all pass
        probs = np.array([0.92, 0.04, 0.02, 0.02], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs)
        assert is_ambiguous is False
        assert reason == ""

    def test_real_world_moderate_confidence_passes(self):
        # Mirrors a real CT scan returning 53%/27%/20%.
        # conf=0.53 ≥ 0.35, margin=0.26 ≥ 0.10, norm_H≈0.922 < 0.95 → all pass.
        # This case was incorrectly rejected by the old threshold (0.75).
        probs = np.array([0.53, 0.27, 0.20], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs)
        assert is_ambiguous is False

    def test_confidence_below_floor_fails(self):
        # conf=0.34 < CONF_THRESHOLD (0.35) → ambiguous
        probs = np.array([0.34, 0.34, 0.32], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs)
        assert is_ambiguous is True
        assert "confidence" in reason

    def test_near_tie_fails_on_margin(self):
        # conf=0.38 ≥ 0.35 ✓ but margin=0.01 < MARGIN_THRESHOLD (0.10) → ambiguous
        probs = np.array([0.38, 0.37, 0.15, 0.10], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs)
        assert is_ambiguous is True
        assert "margin" in reason

    def test_high_normalized_entropy_fails(self):
        # 3-class: conf=0.45 ≥ 0.35, margin=0.10 (exactly at threshold, passes),
        # norm_H≈0.955 > ENTROPY_THRESHOLD (0.95) → ambiguous
        probs = np.array([0.45, 0.35, 0.20], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs)
        assert is_ambiguous is True
        assert "entropy" in reason

    def test_uniform_distribution_fails(self):
        # All equal → conf=0.25 < 0.35 and norm_H=1.0 > 0.95 → ambiguous
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs)
        assert is_ambiguous is True

    def test_reason_lists_failing_criteria(self):
        # Flat distribution fails at least the confidence check
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        _, reason = _assess_prediction_quality(probs)
        assert "confidence" in reason

    def test_very_high_confidence_passes(self):
        # Realistic very high confidence: secondary classes have small but non-zero mass.
        # [1.0, 0.0, ...] is excluded — norm_H = 0 < ENTROPY_FLOOR, and no real model
        # ever produces exactly 1.0; the entropy floor catches that extreme correctly.
        # [0.98, 0.01, 0.006, 0.004]: norm_H ≈ 0.086 > 0.05 → passes floor.
        probs = np.array([0.98, 0.01, 0.006, 0.004], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs)
        assert is_ambiguous is False

    def test_extreme_overconfidence_fails_entropy_floor(self):
        # 4-class: p₁ = 99.97% → norm_H ≈ 0.002 < ENTROPY_FLOOR (0.01).
        # True OOD over-commitment: no well-trained medical model reaches this
        # level of concentration on a valid in-distribution image.
        # (The previous threshold 0.05 incorrectly rejected 99.7% confidence —
        #  a legitimate output on clear pituitary / no-tumor scans.)
        probs = np.array([0.9997, 0.0001, 0.0001, 0.0001], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs)
        assert is_ambiguous is True
        assert "entropy" in reason

    def test_high_confidence_pituitary_prediction_passes(self):
        # Clear pituitary adenoma → model outputs 99.7% on the pituitary class.
        # norm_H ≈ 0.017 > ENTROPY_FLOOR (0.01) → must PASS the quality gate.
        # (pituitary is index 3 in ["glioma","meningioma","notumor","pituitary"])
        probs = np.array([0.001, 0.001, 0.001, 0.997], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs)
        assert is_ambiguous is False

    def test_high_confidence_no_tumor_prediction_passes(self):
        # Clear normal brain scan → model outputs 99.7% on the notumor class.
        # norm_H ≈ 0.017 > ENTROPY_FLOOR (0.01) → must PASS the quality gate.
        # (notumor is index 2 in ["glioma","meningioma","notumor","pituitary"])
        probs = np.array([0.001, 0.001, 0.997, 0.001], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs)
        assert is_ambiguous is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cancer classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyCancer:

    def _arr(self):
        return preprocess_image(_make_image_bytes())

    def _mock_brain(self, probs=(0.92, 0.03, 0.03, 0.02)):
        return _make_mock_model(list(probs))

    def test_brain_high_confidence_glioma(self):
        mock = self._mock_brain((0.92, 0.03, 0.03, 0.02))
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert result["predicted_class"] == "Glioma"

    def test_brain_ambiguous_prediction(self):
        # near-tie: margin=0.02 < MARGIN_THRESHOLD (0.10) → ambiguous → "Invalid Scan"
        mock = self._mock_brain((0.38, 0.36, 0.14, 0.12))
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert result["predicted_class"] == "Invalid Scan"

    def test_brain_low_confidence_prediction(self):
        # conf=0.55: passes all quality checks (conf ≥ 0.35, margin ≥ 0.10)
        # argmax=0 → "glioma" → "Glioma"
        mock = self._mock_brain((0.55, 0.22, 0.14, 0.09))
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert result["predicted_class"] == "Glioma"

    def test_lung_classification(self):
        # IQ-OTH/NCCD classes: ["benign", "malignant", "normal"] (alphabetical)
        # probs [0.05, 0.03, 0.92] → index 2 → "normal" → "Normal Lung"
        mock = _make_mock_model([0.05, 0.03, 0.92])
        with patch.dict(predict_module._models, {"lung": mock}):
            result = classify_cancer(self._arr(), "lung")
        assert result["predicted_class"] == "Normal Lung"

    def test_colon_classification(self):
        # Kather 2016 classes (alphabetical): adipose, complex, debris, empty,
        # lympho, mucosa, stroma, tumor (8 classes, index 7 = tumor)
        probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93]
        mock  = _make_mock_model(probs)
        with patch.dict(predict_module._models, {"colon": mock}):
            result = classify_cancer(self._arr(), "colon")
        assert result["predicted_class"] == "Colorectal Adenocarcinoma"

    def test_predicted_class_is_human_readable(self):
        # The returned predicted_class must use the LABEL_MAPPING display name,
        # not the raw training label (e.g. "Glioma" not "glioma").
        mock = self._mock_brain()
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        raw_labels = set(CLASS_LABELS["brain"])
        assert result["predicted_class"] not in raw_labels, (
            "predicted_class must be the human-readable display name, not a raw training label"
        )

    def test_response_contains_only_predicted_class(self):
        # The response from classify_cancer must contain only predicted_class —
        # no confidence, prediction_status, or all_probabilities fields.
        mock = self._mock_brain()
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert set(result.keys()) == {"predicted_class"}

    # ── Mandatory scenario 1 & 2: class-specific high-confidence correctness ─────

    def test_pituitary_tumor_classified_correctly(self):
        # Scenario 1: pituitary adenoma → model outputs 99.7% on class index 3.
        # With ENTROPY_FLOOR=0.01 this must reach argmax and return "Pituitary Tumor".
        mock = _make_mock_model([0.001, 0.001, 0.001, 0.997])
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert result["predicted_class"] == "Pituitary Tumor"

    def test_no_tumor_classified_correctly(self):
        # Scenario 2: clear normal scan → model outputs 99.7% on class index 2.
        mock = _make_mock_model([0.001, 0.001, 0.997, 0.001])
        with patch.dict(predict_module._models, {"brain": mock}):
            result = classify_cancer(self._arr(), "brain")
        assert result["predicted_class"] == "No Tumor"

    def test_unknown_cancer_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown cancer_type"):
            classify_cancer(self._arr(), "kidney")

    def test_missing_model_raises_runtime_error(self):
        with patch.dict(predict_module._models, {}, clear=True):
            with pytest.raises(RuntimeError, match="not loaded"):
                classify_cancer(self._arr(), "brain")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full pipeline  (predict.predict)
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end tests using mocked models."""

    def _models(self, brain_probs=(0.92, 0.03, 0.03, 0.02)):
        brain = _make_mock_model(list(brain_probs))
        lung  = _make_mock_model([0.05, 0.03, 0.92])
        colon = _make_mock_model([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.93])
        return {"brain": brain, "lung": lung, "colon": colon}

    def test_valid_brain_scan(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["cancer_type"] == "brain"
        assert result["predicted_class"] == "Glioma"

    def test_valid_lung_scan(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(img, "lung")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Normal Lung"

    def test_valid_colon_scan(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(img, "colon")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Colorectal Adenocarcinoma"

    def test_corrupted_image_is_rejected(self):
        corrupt = b"this is not an image at all" * 200
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(corrupt, "brain")
        assert result["status"] == "rejected"
        assert result["reason"] == "invalid_input"

    def test_empty_bytes_rejected(self):
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(b"", "brain")
        assert result["status"] == "rejected"
        assert result["reason"] == "invalid_input"

    def test_invalid_cancer_type_rejected(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(img, "kidney")
        assert result["status"] == "rejected"
        assert result["reason"] == "invalid_cancer_type"

    def test_ambiguous_prediction_returns_success_with_invalid_scan(self):
        img = _make_image_bytes()
        # Flat distribution → fails confidence threshold → "Invalid Scan"
        mods = self._models(brain_probs=(0.25, 0.25, 0.25, 0.25))
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Invalid Scan"

    def test_noise_image_returns_result(self):
        rng   = np.random.default_rng(0)
        noise = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        img   = Image.fromarray(noise)
        buf   = io.BytesIO()
        img.save(buf, format="JPEG")
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(buf.getvalue(), "brain")
        assert result["status"] == "success"

    def test_cancer_type_is_lowercased_and_stripped(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            result = predict_module.predict(img, "  Brain  ")
        assert result["status"] == "success"

    def test_missing_model_returns_error_not_exception(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, {}, clear=True):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "error"
        assert result["reason"] == "model_unavailable"

    def test_domain_screen_rejects_non_medical_image(self):
        # Simulate screen_image_domain flagging a colourful (non-medical) image.
        # The model is never called; the pipeline short-circuits at Stage 2.
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "non_medical_image", "saturation 0.42 exceeds brain limit 0.15"),
            ):
                result = predict_module.predict(img, "brain")
        assert result["status"] == "rejected"
        assert result["reason"] == "non_medical_image"
        assert "saturation" in result["message"]

    def test_domain_screen_rejects_cross_modality_image(self):
        # Simulate screen_image_domain detecting a CT/MRI sent to the colon model
        # (Signal 2: saturation below the colon floor).
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "cross_modality", "saturation 0.02 is below colon minimum 0.05"),
            ):
                result = predict_module.predict(img, "colon")
        assert result["status"] == "rejected"
        assert result["reason"] == "cross_modality"
        assert "saturation" in result["message"]


# ─────────────────────────────────────────────────────────────────────────────
# 9. Robust validation — tolerance and edge-case coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestRobustValidation:
    """
    Validates the "strict with invalid, tolerant with valid" design principle.

    Tests 1–3  Valid scans accepted by their correct model (covered in TestFullPipeline;
               referenced here for clarity only — see that class).

    Test  4    Moderate-quality scan returns low_confidence, NOT rejected.
    Tests 5–7  Clearly non-medical inputs are rejected (logo, photo, noise).
    Test  8    Blurry/uncertain scan returns ambiguous or low_confidence, NOT rejected.
    Test  9    CT submitted to colon model — caught by Signal 2 (saturation floor).
    Test 10    Histopathology submitted to lung model — flagged ambiguous by quality gate,
               NOT a hard rejection (Signal 1 catches high-saturation H&E; low-saturation
               H&E passes the domain screen and is handled by the softmax gate).
    Test 11    Cross-model consistency — the correct model is confident; other models
               return low_confidence or ambiguous on the same image.
    """

    def _make_models(
        self,
        brain_probs=(0.92, 0.03, 0.03, 0.02),
        lung_probs=(0.05, 0.03, 0.92),
        colon_probs=None,
    ):
        if colon_probs is None:
            colon_probs = [0.01] * 7 + [0.93]
        return {
            "brain": _make_mock_model(list(brain_probs)),
            "lung" : _make_mock_model(list(lung_probs)),
            "colon": _make_mock_model(list(colon_probs)),
        }

    # ── Test 4: Moderate-quality scan (valid but low confidence) ─────────────

    def test_moderate_quality_lung_scan_accepted(self):
        # A real-world CT that returns 55% confidence: passes quality gate
        # (conf ≥ 0.35, margin ≥ 0.10). Must return "success", NOT rejected.
        # probs [0.10, 0.55, 0.35] → argmax=1 → "malignant" → "Malignant Lung Cancer"
        mods = self._make_models(lung_probs=[0.10, 0.55, 0.35])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "lung")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Malignant Lung Cancer"

    def test_moderate_quality_brain_scan_accepted(self):
        # Brain MRI returning 58% on top class: low-confidence but valid.
        # probs [0.58, 0.22, 0.12, 0.08] → argmax=0 → "glioma" → "Glioma"
        mods = self._make_models(brain_probs=[0.58, 0.22, 0.12, 0.08])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Glioma"

    # ── Tests 5–7: Clearly non-medical images rejected ───────────────────────

    def test_logo_image_rejected_non_medical(self):
        # High-saturation colourful image → Signal 1 rejects before model runs.
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._make_models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "non_medical_image", "saturation 0.72 exceeds brain limit 0.15"),
            ):
                result = predict_module.predict(img, "brain")
        assert result["status"] == "rejected"
        assert result["reason"] == "non_medical_image"

    def test_random_photo_rejected_non_medical(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._make_models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "non_medical_image", "saturation 0.45 exceeds lung limit 0.15"),
            ):
                result = predict_module.predict(img, "lung")
        assert result["status"] == "rejected"
        assert result["reason"] == "non_medical_image"

    def test_noise_image_returns_result_not_exception(self):
        # Random noise passes the domain screen (gray, low saturation) and
        # reaches the model.  The model may return ambiguous or a prediction;
        # the key invariant is that the pipeline returns a structured dict.
        rng   = np.random.default_rng(42)
        noise = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        buf   = io.BytesIO()
        Image.fromarray(noise).save(buf, format="JPEG")
        with patch.dict(predict_module._models, self._make_models()):
            result = predict_module.predict(buf.getvalue(), "brain")
        assert result["status"] in ("success", "rejected")
        assert "reason" in result or "predicted_class" in result

    # ── Test 8: Blurry / uncertain scan ──────────────────────────────────────

    def test_blurry_scan_returns_ambiguous_not_rejected(self):
        # A blurry scan produces an uncertain softmax distribution.
        # Low confidence (0.38) and small margin (0.06 < 0.07) fail the quality gate
        # → "Invalid Scan", but status="success" (not "rejected").
        # The system must NOT hard-reject uncertain-but-medical images.
        # Probs: [0.38, 0.32, 0.20, 0.10] → margin = 0.38 - 0.32 = 0.06 < 0.07
        mods = self._make_models(brain_probs=[0.38, 0.32, 0.20, 0.10])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"           # pipeline completed
        assert result["predicted_class"] == "Invalid Scan"

    def test_low_contrast_scan_with_clear_margin_accepted(self):
        # A low-contrast scan can still produce a clear winner.
        # [0.50, 0.28, 0.22]: conf=0.50 (≥0.35 ✓), margin=0.22 (≥0.10 ✓),
        # norm_H≈0.943 (<0.95 ✓) → accepted.
        # probs → argmax=0 → "benign" → "Benign Lung Lesion"
        mods = self._make_models(lung_probs=[0.50, 0.28, 0.22])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "lung")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Benign Lung Lesion"

    # ── Test 9: CT submitted to colon model ──────────────────────────────────

    def test_ct_sent_to_colon_not_confidently_classified(self):
        # A CT scan (near-grayscale, saturation ≈ 0.02) sent to the colon model
        # is caught by Signal 2 (saturation floor 0.05) → cross_modality rejection.
        # It must NOT produce a confident colon tissue prediction.
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._make_models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "cross_modality", "saturation 0.02 is below colon minimum 0.05"),
            ):
                result = predict_module.predict(img, "colon")
        assert result["status"] == "rejected"
        assert result["reason"] == "cross_modality"

    # ── Test 10: Histopathology submitted to lung model ───────────────────────

    def test_histopathology_sent_to_lung_not_confidently_classified(self):
        # Lightly-stained H&E (saturation ≈ 0.10 < 0.15 ceiling) passes Signal 1.
        # The lung model, applied to histopathology, produces a flat distribution
        # → quality gate returns "Invalid Scan".
        flat_probs = [0.34, 0.33, 0.33]  # near-uniform → conf < 0.35 → ambiguous
        mods = self._make_models(lung_probs=flat_probs)
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "lung")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Invalid Scan"

    def test_high_saturation_histopathology_sent_to_lung_domain_rejected(self):
        # Normally-stained H&E (saturation ≈ 0.25 > 0.15 lung ceiling) is caught
        # by Signal 1 before reaching the model.
        img = _make_image_bytes()
        with patch.dict(predict_module._models, self._make_models()):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "non_medical_image", "saturation 0.25 exceeds lung limit 0.15"),
            ):
                result = predict_module.predict(img, "lung")
        assert result["status"] == "rejected"
        assert result["reason"] == "non_medical_image"

    # ── Test 11: Cross-model consistency ──────────────────────────────────────

    def test_correct_model_confident_others_ambiguous(self):
        # When a brain MRI is routed to its correct model it returns a class name.
        # The same image routed to the lung model (wrong modality) produces a flat
        # distribution → "Invalid Scan".  Each model is queried independently.
        brain_mods = self._make_models(brain_probs=[0.92, 0.04, 0.02, 0.02])
        lung_mods  = self._make_models(lung_probs=[0.34, 0.33, 0.33])   # near-uniform
        img = _make_image_bytes()

        with patch.dict(predict_module._models, brain_mods):
            brain_result = predict_module.predict(img, "brain")
        with patch.dict(predict_module._models, lung_mods):
            lung_result  = predict_module.predict(img, "lung")

        assert brain_result["predicted_class"] == "Glioma"
        assert lung_result["predicted_class"]  == "Invalid Scan"

    def test_wrong_model_never_returns_confident_for_ambiguous_input(self):
        # An ambiguous input (near-uniform distribution) sent to the colon model
        # must return "Invalid Scan", not a tissue class.
        ambiguous_colon = [0.13, 0.13, 0.13, 0.13, 0.12, 0.12, 0.12, 0.12]
        mods = self._make_models(colon_probs=ambiguous_colon)
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "colon")
        assert result["predicted_class"] == "Invalid Scan"

    # ── Mandatory scenarios 1–6: full pipeline ────────────────────────────────

    def test_scenario_1_pituitary_tumor_pipeline(self):
        # Scenario 1: pituitary adenoma — 99.7% confidence must reach argmax,
        # not be rejected by entropy floor.
        mods = self._make_models(brain_probs=[0.001, 0.001, 0.001, 0.997])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Pituitary Tumor"

    def test_scenario_2_no_tumor_pipeline(self):
        # Scenario 2: clear normal scan — 99.7% confidence must not be rejected.
        mods = self._make_models(brain_probs=[0.001, 0.001, 0.997, 0.001])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "No Tumor"

    def test_scenario_3_glioma_unchanged(self):
        # Scenario 3a: glioma — high confidence, should be unaffected.
        mods = self._make_models(brain_probs=[0.92, 0.04, 0.02, 0.02])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Glioma"

    def test_scenario_3_meningioma_unchanged(self):
        # Scenario 3b: meningioma — should be unaffected.
        mods = self._make_models(brain_probs=[0.04, 0.90, 0.03, 0.03])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Meningioma"

    def test_scenario_4_low_confidence_tumor_returns_class(self):
        # Scenario 4: low-confidence scan (55% on pituitary, clear margin) —
        # passes conf (≥0.35) and margin (≥0.10), must return class not "Invalid Scan".
        mods = self._make_models(brain_probs=[0.10, 0.15, 0.20, 0.55])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Pituitary Tumor"

    def test_scenario_5_clear_no_tumor_not_rejected(self):
        # Scenario 5: explicit alias for scenario 2 — clear no-tumor must not
        # be rejected regardless of how high the confidence is.
        mods = self._make_models(brain_probs=[0.002, 0.001, 0.996, 0.001])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "No Tumor"

    def test_scenario_6_slightly_ambiguous_scan_handled_gracefully(self):
        # Scenario 6: near-tie (margin=0.06 < 0.07) → quality gate fires →
        # "Invalid Scan" returned.  Status is still "success" (not "rejected").
        # The system must NOT raise an exception or return an error status.
        mods = self._make_models(brain_probs=[0.40, 0.34, 0.14, 0.12])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Invalid Scan"

    # ── notumor (No Tumor) — moderate-confidence acceptance ──────────────────

    def test_notumor_with_small_but_decisive_margin_passes(self):
        # notumor (index 2) at 45% with margin=0.08 (> new threshold 0.07).
        # The masoudnickparvar dataset has ~2× fewer notumor images than tumour
        # classes; the model naturally produces smaller margins on this class.
        # This was incorrectly rejected by the old MARGIN_THRESHOLD=0.10.
        mods = self._make_models(brain_probs=[0.15, 0.22, 0.45, 0.18])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        # margin = 0.45 - 0.22 = 0.23 > 0.07 → accepted
        assert result["predicted_class"] == "No Tumor"

    def test_notumor_with_margin_exactly_at_new_threshold_passes(self):
        # margin = 0.08 > 0.07 → must pass; was previously caught by 0.10 threshold.
        # [0.18, 0.20, 0.43, 0.19] → argmax=2 (notumor), margin = 0.43-0.20 = 0.23
        # Use [0.19, 0.35, 0.43, 0.03] → margin = 0.43 - 0.35 = 0.08 (borderline)
        mods = self._make_models(brain_probs=[0.19, 0.35, 0.43, 0.03])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        # margin = 0.43 - 0.35 = 0.08 > 0.07 → no longer rejected
        assert result["predicted_class"] == "No Tumor"

    def test_notumor_true_near_tie_still_rejected(self):
        # margin = 0.06 < 0.07 → quality gate must still fire even for notumor.
        # [0.21, 0.35, 0.41, 0.03] → margin = 0.41 - 0.35 = 0.06 → "Invalid Scan"
        mods = self._make_models(brain_probs=[0.21, 0.35, 0.41, 0.03])
        img  = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            result = predict_module.predict(img, "brain")
        assert result["status"] == "success"
        assert result["predicted_class"] == "Invalid Scan"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Colon-specific quality gate
# ─────────────────────────────────────────────────────────────────────────────

class TestColonQualityGate:
    """
    Tests for the colon-specific lenient quality gate thresholds.

    The Kather 2016 8-class model covers tissue types that include low-information
    classes (empty background, sparse stroma, adipose, cellular debris).  These
    valid inputs naturally produce weaker softmax signals than tumour classes:
        conf ≈ 20–30%  (above random baseline 12.5%, below global floor 35%)
        margin ≈ 2–6%  (small but decisive for 8 classes)
        norm_H ≈ 0.96–0.98  (high entropy from multi-class ambiguity)

    Colon thresholds: CONF=0.20, MARGIN=0.02, ENTROPY_CEILING=0.99.
    Global thresholds (brain/lung): CONF=0.35, MARGIN=0.07, ENTROPY_CEILING=0.95.
    ENTROPY_FLOOR=0.01 is NOT overridden — OOD detection is modality-invariant.
    """

    # ── Unit: _assess_prediction_quality with cancer_type="colon" ────────────

    def test_sparse_stroma_distribution_passes_colon_gate(self):
        # Stroma/debris distribution: conf=0.28 > 0.20, margin=0.04 > 0.02,
        # norm_H ≈ 0.858 < 0.99 → passes colon gate.
        # The same distribution fails the global gate: conf=0.28 < 0.35.
        probs = np.array([0.28, 0.24, 0.20, 0.10, 0.08, 0.05, 0.03, 0.02], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is False
        assert reason == ""

    def test_low_cellularity_distribution_passes_colon_gate(self):
        # Very uncertain but above random: conf=0.24 > 0.20, margin=0.04 > 0.02,
        # norm_H ≈ 0.907 < 0.99 → passes.
        probs = np.array([0.24, 0.20, 0.18, 0.15, 0.11, 0.05, 0.04, 0.03], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is False

    def test_high_entropy_colon_distribution_passes(self):
        # norm_H ≈ 0.961 — rejected by global entropy ceiling (0.95) but
        # accepted by colon ceiling (0.99).
        # conf=0.20 (exactly at colon floor), margin=0.03 > 0.02 → passes.
        probs = np.array([0.20, 0.17, 0.15, 0.14, 0.12, 0.10, 0.08, 0.04], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is False

    def test_colon_truly_random_still_rejected(self):
        # Uniform distribution: conf=0.125 < 0.20 → rejected even with colon gate.
        probs = np.array([0.125] * 8, dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is True
        assert "confidence" in reason

    def test_colon_near_random_still_rejected(self):
        # conf=0.15 < 0.20 → rejected.
        probs = np.array([0.15, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.10], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is True

    def test_colon_minimal_margin_still_rejected(self):
        # conf=0.25 > 0.20 ✓ but margin=0.01 < 0.02 → rejected.
        probs = np.array([0.25, 0.24, 0.13, 0.13, 0.10, 0.08, 0.04, 0.03], dtype=np.float32)
        is_ambiguous, _ = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is True

    def test_same_distribution_rejected_for_brain_model(self):
        # The uncertain stroma distribution that passes the colon gate must be
        # rejected when routed through the brain model (global thresholds).
        # conf=0.28 < CONF_THRESHOLD (0.35) → ambiguous.
        probs = np.array([0.28, 0.24, 0.20, 0.10, 0.08, 0.05, 0.03, 0.02], dtype=np.float32)
        is_ambiguous, reason = _assess_prediction_quality(probs, cancer_type="brain")
        assert is_ambiguous is True
        assert "confidence" in reason

    def test_colon_oob_over_commitment_rejected_by_entropy_floor(self):
        # p₁ ≈ 99.8% → norm_H ≈ 0.0087 < ENTROPY_FLOOR (0.01).
        # ENTROPY_FLOOR is not overridden for colon; fires as for any modality.
        probs = np.array(
            [0.998, 0.0003, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0006],
            dtype=np.float32,
        )
        is_ambiguous, reason = _assess_prediction_quality(probs, cancer_type="colon")
        assert is_ambiguous is True
        assert "entropy" in reason

    # ── Integration: classify_cancer uses colon gate internally ──────────────

    def test_stroma_like_prediction_accepted_by_colon_pipeline(self):
        # Low-confidence stroma distribution must reach argmax, not "Invalid Scan".
        # probs[0] is largest → index 0 → "adipose" → "Adipose Tissue"
        probs = [0.28, 0.24, 0.20, 0.10, 0.08, 0.05, 0.03, 0.02]
        mock  = _make_mock_model(probs)
        arr   = preprocess_image(_make_image_bytes())
        with patch.dict(predict_module._models, {"colon": mock}):
            result = classify_cancer(arr, "colon")
        assert result["predicted_class"] != "Invalid Scan"
        assert result["predicted_class"] == "Adipose Tissue"

    def test_empty_background_prediction_accepted_by_colon_pipeline(self):
        # Very high-entropy distribution (norm_H ≈ 0.907) still passes colon gate.
        # argmax = index 0 → "adipose" → "Adipose Tissue"
        probs = [0.24, 0.20, 0.18, 0.15, 0.11, 0.05, 0.04, 0.03]
        mock  = _make_mock_model(probs)
        arr   = preprocess_image(_make_image_bytes())
        with patch.dict(predict_module._models, {"colon": mock}):
            result = classify_cancer(arr, "colon")
        assert result["predicted_class"] != "Invalid Scan"

    def test_colon_near_random_rejected_in_full_pipeline(self):
        # conf=0.15 < 0.20 even with colon thresholds → "Invalid Scan".
        probs = [0.15, 0.14, 0.13, 0.13, 0.12, 0.12, 0.11, 0.10]
        mock  = _make_mock_model(probs)
        arr   = preprocess_image(_make_image_bytes())
        with patch.dict(predict_module._models, {"colon": mock}):
            result = classify_cancer(arr, "colon")
        assert result["predicted_class"] == "Invalid Scan"
