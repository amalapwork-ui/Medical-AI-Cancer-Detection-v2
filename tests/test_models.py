"""
Model specification tests — Medical AI Cancer Detection.

Verifies that model routing, output shapes, class counts, and the
model registry behave correctly for each cancer type.

All tests use mocked models: no GPU, no .keras files, no TensorFlow install.

Run with:
    pytest tests/test_models.py -v
"""

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub tensorflow before any app import.
# keras.Model must be present: predict.py has a module-level annotation
# `_models: dict[str, tf.keras.Model] = {}` evaluated at import time.
tf_stub = types.ModuleType("tensorflow")
tf_stub.keras = types.ModuleType("tensorflow.keras")
tf_stub.keras.Model = MagicMock()
tf_stub.keras.models = types.ModuleType("tensorflow.keras.models")
tf_stub.keras.models.load_model = MagicMock()
sys.modules.setdefault("tensorflow", tf_stub)
sys.modules.setdefault("tensorflow.keras", tf_stub.keras)
sys.modules.setdefault("tensorflow.keras.models", tf_stub.keras.models)

# Stub cv2 so predict.py imports cleanly
cv2_stub = types.ModuleType("cv2")
_clahe_stub = MagicMock()
_clahe_stub.apply = lambda x: x
cv2_stub.createCLAHE = MagicMock(return_value=_clahe_stub)
cv2_stub.COLOR_BGR2RGB = 4
cv2_stub.INTER_LINEAR = 1
sys.modules.setdefault("cv2", cv2_stub)

from backend.app import predict as predict_module
from backend.app.predict import classify_cancer, load_models, preprocess_image
from backend.app.config import CLASS_LABELS, IMAGE_SIZE, MODEL_PATHS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_bytes(width: int = 300, height: int = 300) -> bytes:
    img = Image.new("RGB", (width, height), (128, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_model(num_classes: int, top_class: int = 0) -> MagicMock:
    """Return a mock model that outputs a one-hot-ish probability vector."""
    probs = [0.01] * num_classes
    probs[top_class] = 1.0 - 0.01 * (num_classes - 1)
    m = MagicMock()
    m.predict = MagicMock(return_value=np.array([probs], dtype=np.float32))
    return m


def _input_array() -> np.ndarray:
    return preprocess_image(_make_image_bytes())


# ─────────────────────────────────────────────────────────────────────────────
# 1. CLASS_LABELS specification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassLabelSpec:
    """
    Verify that CLASS_LABELS in config.py matches what each training script
    produces (Keras alphabetical sort of class folder names).
    """

    def test_brain_has_four_classes(self):
        assert len(CLASS_LABELS["brain"]) == 4

    def test_brain_class_order(self):
        assert CLASS_LABELS["brain"] == ["glioma", "meningioma", "notumor", "pituitary"]

    def test_lung_has_three_classes(self):
        assert len(CLASS_LABELS["lung"]) == 3

    def test_lung_class_order_alphabetical(self):
        # IQ-OTH/NCCD: benign, malignant, normal (alphabetical)
        assert CLASS_LABELS["lung"] == ["benign", "malignant", "normal"]

    def test_colon_has_eight_classes(self):
        assert len(CLASS_LABELS["colon"]) == 8

    def test_colon_class_order_alphabetical(self):
        # Kather 2016 8 tissue types in alphabetical order
        expected = ["adipose", "complex", "debris", "empty",
                    "lympho", "mucosa", "stroma", "tumor"]
        assert CLASS_LABELS["colon"] == expected

    def test_all_cancer_types_present(self):
        assert set(CLASS_LABELS.keys()) == {"brain", "lung", "colon"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model output shape per cancer type
# ─────────────────────────────────────────────────────────────────────────────

class TestModelOutputShape:
    """
    Verify that classify_cancer() returns all_probabilities with the correct
    number of entries for each cancer type.
    """

    def _classify(self, cancer_type: str) -> dict:
        n = len(CLASS_LABELS[cancer_type])
        mock = _mock_model(n)
        with patch.dict(predict_module._models, {cancer_type: mock}):
            return classify_cancer(_input_array(), cancer_type)

    def test_brain_returns_four_probabilities(self):
        result = self._classify("brain")
        assert len(result["all_probabilities"]) == 4

    def test_lung_returns_three_probabilities(self):
        result = self._classify("lung")
        assert len(result["all_probabilities"]) == 3

    def test_colon_returns_eight_probabilities(self):
        result = self._classify("colon")
        assert len(result["all_probabilities"]) == 8

    def test_probabilities_sum_to_one(self):
        for cancer_type in ("brain", "lung", "colon"):
            result = self._classify(cancer_type)
            total = sum(result["all_probabilities"].values())
            assert abs(total - 1.0) < 0.02, (
                f"{cancer_type}: probabilities sum to {total}, expected ~1.0"
            )

    def test_confidence_is_percentage_range(self):
        for cancer_type in ("brain", "lung", "colon"):
            result = self._classify(cancer_type)
            assert 0.0 <= result["confidence"] <= 100.0, (
                f"{cancer_type}: confidence={result['confidence']} out of [0,100]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correct class label routing per model
# ─────────────────────────────────────────────────────────────────────────────

class TestClassLabelRouting:
    """
    Verify that argmax index maps to the correct human-readable label for each model.
    The index→label mapping must match config.py CLASS_LABELS order exactly.
    """

    def _predict_class(self, cancer_type: str, top_idx: int) -> str:
        n = len(CLASS_LABELS[cancer_type])
        mock = _mock_model(n, top_class=top_idx)
        with patch.dict(predict_module._models, {cancer_type: mock}):
            result = classify_cancer(_input_array(), cancer_type)
        return result["predicted_class"]

    # Brain: glioma=0, meningioma=1, notumor=2, pituitary=3
    def test_brain_index0_is_glioma(self):
        assert self._predict_class("brain", 0) == "Glioma"

    def test_brain_index1_is_meningioma(self):
        assert self._predict_class("brain", 1) == "Meningioma"

    def test_brain_index2_is_notumor(self):
        assert self._predict_class("brain", 2) == "No Tumor"

    def test_brain_index3_is_pituitary(self):
        assert self._predict_class("brain", 3) == "Pituitary Tumor"

    # Lung: benign=0, malignant=1, normal=2
    def test_lung_index0_is_benign(self):
        assert self._predict_class("lung", 0) == "Benign Lung Lesion"

    def test_lung_index1_is_malignant(self):
        assert self._predict_class("lung", 1) == "Malignant Lung Cancer"

    def test_lung_index2_is_normal(self):
        assert self._predict_class("lung", 2) == "Normal Lung"

    # Colon: adipose=0, complex=1, debris=2, empty=3, lympho=4, mucosa=5, stroma=6, tumor=7
    def test_colon_index0_is_adipose(self):
        assert self._predict_class("colon", 0) == "Adipose Tissue"

    def test_colon_index6_is_stroma(self):
        assert self._predict_class("colon", 6) == "Cancer-Associated Stroma"

    def test_colon_index7_is_tumor(self):
        assert self._predict_class("colon", 7) == "Colorectal Adenocarcinoma"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prediction status tiers
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionStatusTiers:
    """
    Verify the three prediction status tiers:
      conf ≥ 0.65 and quality gate passes → confident
      0.35 ≤ conf < 0.65, quality gate passes → low_confidence
      quality gate fails (conf < 0.35, margin < 0.10, or norm_H > 0.95) → ambiguous
    """

    def _classify_with_probs(self, probs: list) -> dict:
        mock = MagicMock()
        mock.predict = MagicMock(
            return_value=np.array([probs], dtype=np.float32)
        )
        with patch.dict(predict_module._models, {"brain": mock}):
            return classify_cancer(_input_array(), "brain")

    def test_confident_tier(self):
        # conf=0.90 ≥ CONFIDENCE_LOW (0.65), all gate criteria pass
        probs  = [0.90, 0.05, 0.03, 0.02]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "confident"

    def test_low_confidence_tier(self):
        # conf=0.55 < CONFIDENCE_LOW (0.65) but ≥ CONF_THRESHOLD (0.35)
        # margin=0.33 ≥ 0.10, norm_H≈0.833 < 0.95 → quality gate passes → low_confidence
        probs  = [0.55, 0.22, 0.14, 0.09]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "low_confidence"

    def test_ambiguous_near_tie(self):
        # margin=0.05 < MARGIN_THRESHOLD (0.10) → gate fails → ambiguous
        probs  = [0.38, 0.33, 0.18, 0.11]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "ambiguous"
        assert result["predicted_class"] == "Invalid Scan"

    def test_ambiguous_below_confidence_floor(self):
        # conf=0.30 < CONF_THRESHOLD (0.35) → gate fails → ambiguous
        probs  = [0.30, 0.28, 0.24, 0.18]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "ambiguous"
        assert result["predicted_class"] == "Invalid Scan"

    def test_above_confident_boundary(self):
        # conf=0.70 > CONFIDENCE_LOW (0.65) → "confident"
        # margin=0.53 ≥ 0.10, norm_H≈0.707 < 0.95 → gate passes
        # (0.65 itself is avoided: np.float32(0.65) rounds to 0.6499... < 0.65)
        probs  = [0.70, 0.17, 0.08, 0.05]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "confident"

    def test_just_below_confident_boundary(self):
        # conf=0.64 just below CONFIDENCE_LOW (0.65) → "low_confidence"
        probs  = [0.64, 0.20, 0.10, 0.06]
        result = self._classify_with_probs(probs)
        assert result["prediction_status"] == "low_confidence"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model registry behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    """
    Verify load_models() handles missing files gracefully and the registry
    helpers (loaded_models, missing_models) report accurately.
    """

    def test_load_models_skips_nonexistent_files(self):
        # No .keras files exist in test environment — load_models must not raise
        with patch.dict(predict_module._models, {}, clear=True):
            try:
                load_models()
            except Exception as exc:
                pytest.fail(f"load_models() raised unexpectedly: {exc}")

    def test_missing_model_raises_runtime_error_in_classify(self):
        with patch.dict(predict_module._models, {}, clear=True):
            with pytest.raises(RuntimeError, match="not loaded"):
                classify_cancer(_input_array(), "brain")

    def test_unknown_cancer_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown cancer_type"):
            classify_cancer(_input_array(), "prostate")

    def test_model_predict_called_once_per_request(self):
        mock = _mock_model(4)
        with patch.dict(predict_module._models, {"brain": mock}):
            classify_cancer(_input_array(), "brain")
        assert mock.predict.call_count == 1

    def test_all_cancer_types_routed_to_correct_model(self):
        """Each cancer_type must route to its own model, not another model's."""
        brain_mock = _mock_model(4)
        lung_mock  = _mock_model(3)
        colon_mock = _mock_model(8)
        all_mocks  = {"brain": brain_mock, "lung": lung_mock, "colon": colon_mock}

        with patch.dict(predict_module._models, all_mocks):
            classify_cancer(_input_array(), "brain")
            classify_cancer(_input_array(), "lung")
            classify_cancer(_input_array(), "colon")

        assert brain_mock.predict.call_count == 1
        assert lung_mock.predict.call_count  == 1
        assert colon_mock.predict.call_count == 1

    def test_model_paths_defined_for_all_expected_models(self):
        expected = {"brain", "lung", "colon"}
        assert set(MODEL_PATHS.keys()) == expected

    def test_model_paths_are_under_backend_models_dir(self):
        for name, path in MODEL_PATHS.items():
            assert "models" in str(path), (
                f"MODEL_PATHS['{name}'] = {path} does not point into backend/models/"
            )
            assert path.suffix == ".keras", (
                f"MODEL_PATHS['{name}'] does not end in .keras"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Input array shape contract
# ─────────────────────────────────────────────────────────────────────────────

class TestInputArrayContract:
    """
    Verify that preprocess_image always produces the shape and dtype that
    models expect, regardless of input image characteristics.
    """

    def _shape(self, **kwargs) -> tuple:
        data = _make_image_bytes(**kwargs)
        return preprocess_image(data).shape

    def test_standard_image_shape(self):
        assert self._shape() == (1, 224, 224, 3)

    def test_very_small_image_shape(self):
        assert self._shape(width=16, height=16) == (1, 224, 224, 3)

    def test_very_large_image_shape(self):
        assert self._shape(width=1024, height=1024) == (1, 224, 224, 3)

    def test_non_square_image_shape(self):
        assert self._shape(width=1024, height=512) == (1, 224, 224, 3)

    def test_output_dtype_is_float32(self):
        arr = preprocess_image(_make_image_bytes())
        assert arr.dtype == np.float32

    def test_pixel_values_not_normalised(self):
        # include_preprocessing=True: pass raw [0,255]; backbone normalises internally
        arr = preprocess_image(_make_image_bytes())
        assert arr.max() > 1.0, (
            "Pixel values appear pre-normalised to [0,1]. "
            "Models expect raw [0,255] float32."
        )
