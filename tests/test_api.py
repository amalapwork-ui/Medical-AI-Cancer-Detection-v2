"""
FastAPI integration tests — Medical AI Cancer Detection.

Tests the HTTP layer via FastAPI's TestClient (no running server needed).
Models are mocked so tests run without GPU or .keras files.

Run with:
    pytest tests/test_api.py -v
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
# keras.Model is needed at import time: predict.py has a module-level annotation
# `_models: dict[str, tf.keras.Model] = {}` which Python evaluates immediately.
tf_stub = types.ModuleType("tensorflow")
tf_stub.keras = types.ModuleType("tensorflow.keras")
tf_stub.keras.Model = MagicMock()
tf_stub.keras.models = types.ModuleType("tensorflow.keras.models")
tf_stub.keras.models.load_model = MagicMock()
sys.modules.setdefault("tensorflow", tf_stub)
sys.modules.setdefault("tensorflow.keras", tf_stub.keras)
sys.modules.setdefault("tensorflow.keras.models", tf_stub.keras.models)

# Also stub cv2 so predict.py imports cleanly without opencv installed.
# Default HSV: saturation uint8=20 → mean ≈ 0.08.  Passes all per-modality
# saturation checks: below brain/lung ceiling (0.15), above colon floor (0.05),
# and below colon ceiling (0.50).
cv2_stub = types.ModuleType("cv2")
_clahe_stub = MagicMock()
_clahe_stub.apply = lambda x: x  # identity — CLAHE stub returns channel unchanged
_default_hsv = np.zeros((224, 224, 3), dtype=np.uint8)
_default_hsv[:, :, 1] = 20
cv2_stub.createCLAHE   = MagicMock(return_value=_clahe_stub)
cv2_stub.cvtColor      = MagicMock(return_value=_default_hsv)
cv2_stub.COLOR_RGB2HSV = 40
cv2_stub.COLOR_BGR2RGB = 4
cv2_stub.INTER_LINEAR  = 1
sys.modules.setdefault("cv2", cv2_stub)

from fastapi.testclient import TestClient  # noqa: E402
from backend.app import predict as predict_module  # noqa: E402
from backend.app.main import app  # noqa: E402

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_image_bytes(width: int = 300, height: int = 300, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_mock_models(brain_probs: tuple = (0.92, 0.03, 0.03, 0.02)) -> dict:
    brain = MagicMock()
    brain.predict = MagicMock(return_value=np.array([list(brain_probs)], dtype=np.float32))
    lung  = MagicMock()
    lung.predict  = MagicMock(return_value=np.array([[0.05, 0.03, 0.92]], dtype=np.float32))
    colon = MagicMock()
    colon.predict = MagicMock(
        return_value=np.array([[0.01] * 7 + [0.93]], dtype=np.float32)
    )
    return {"brain": brain, "lung": lung, "colon": colon}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_lists_loaded_and_missing_models(self):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "models_missing" in data
        assert isinstance(data["models_loaded"], list)
        assert isinstance(data["models_missing"], list)

    def test_health_degraded_when_no_models_loaded(self):
        with patch.dict(predict_module._models, {}, clear=True):
            response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"

    def test_health_ok_when_all_models_loaded(self):
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods, clear=True):
            response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. /predict — content type validation (HTTP 415)
# ─────────────────────────────────────────────────────────────────────────────

class TestContentTypeValidation:

    def test_jpeg_accepted(self):
        img  = _make_image_bytes()
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            response = client.post(
                "/predict",
                files={"file": ("img.jpg", img, "image/jpeg")},
                data={"cancer_type": "brain"},
            )
        assert response.status_code == 200

    def test_png_accepted(self):
        img  = _make_image_bytes(fmt="PNG")
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            response = client.post(
                "/predict",
                files={"file": ("img.png", img, "image/png")},
                data={"cancer_type": "brain"},
            )
        assert response.status_code == 200

    def test_pdf_rejected_415(self):
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-content", "application/pdf")},
            data={"cancer_type": "brain"},
        )
        assert response.status_code == 415

    def test_text_rejected_415(self):
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"plain text", "text/plain")},
            data={"cancer_type": "brain"},
        )
        assert response.status_code == 415


# ─────────────────────────────────────────────────────────────────────────────
# 3. /predict — successful prediction response shape
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictResponseShape:

    def _post(self, cancer_type: str = "brain", mods: dict = None):
        if mods is None:
            mods = _make_mock_models()
        img = _make_image_bytes()
        with patch.dict(predict_module._models, mods):
            return client.post(
                "/predict",
                files={"file": ("img.jpg", img, "image/jpeg")},
                data={"cancer_type": cancer_type},
            )

    def test_status_200(self):
        assert self._post().status_code == 200

    def test_success_status_in_body(self):
        data = self._post().json()
        assert data["status"] == "success"

    def test_required_fields_present(self):
        data = self._post().json()
        for field in ("predicted_class", "confidence", "prediction_status",
                      "all_probabilities", "cancer_type"):
            assert field in data, f"Missing field: {field}"

    def test_confidence_is_percentage(self):
        data = self._post().json()
        assert 0.0 <= data["confidence"] <= 100.0

    def test_all_probabilities_is_dict(self):
        data = self._post().json()
        assert isinstance(data["all_probabilities"], dict)

    def test_prediction_status_valid_value(self):
        data = self._post().json()
        assert data["prediction_status"] in ("confident", "low_confidence", "ambiguous")

    def test_brain_returns_four_classes(self):
        data = self._post(cancer_type="brain").json()
        assert len(data["all_probabilities"]) == 4

    def test_lung_returns_three_classes(self):
        data = self._post(cancer_type="lung").json()
        assert len(data["all_probabilities"]) == 3

    def test_colon_returns_eight_classes(self):
        data = self._post(cancer_type="colon").json()
        assert len(data["all_probabilities"]) == 8

    def test_brain_predicted_class_is_glioma(self):
        data = self._post(cancer_type="brain").json()
        assert data["predicted_class"] == "Glioma"

    def test_cancer_type_echoed_in_response(self):
        data = self._post(cancer_type="brain").json()
        assert data["cancer_type"] == "brain"

    def test_ambiguous_prediction_returns_invalid_scan(self):
        # Flat distribution → confidence < threshold → "Invalid Scan"
        mods = _make_mock_models(brain_probs=(0.25, 0.25, 0.25, 0.25))
        data = self._post(cancer_type="brain", mods=mods).json()
        assert data["status"] == "success"
        assert data["predicted_class"] == "Invalid Scan"
        assert data["prediction_status"] == "ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
# 4. /predict — rejection cases
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictRejections:

    def _post(self, img_bytes: bytes, cancer_type: str = "brain",
              content_type: str = "image/jpeg", mods: dict = None):
        if mods is None:
            mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            return client.post(
                "/predict",
                files={"file": ("img.jpg", img_bytes, content_type)},
                data={"cancer_type": cancer_type},
            )

    def test_empty_bytes_returns_rejected(self):
        data = self._post(b"").json()
        assert data["status"] == "rejected"
        assert data["reason"] == "invalid_input"

    def test_corrupted_bytes_returns_rejected(self):
        data = self._post(b"not-an-image" * 200).json()
        assert data["status"] == "rejected"
        assert data["reason"] == "invalid_input"

    def test_invalid_cancer_type_returns_rejected(self):
        img  = _make_image_bytes()
        data = self._post(img, cancer_type="kidney").json()
        assert data["status"] == "rejected"
        assert data["reason"] == "invalid_cancer_type"

    def test_missing_model_returns_error(self):
        img = _make_image_bytes()
        with patch.dict(predict_module._models, {}, clear=True):
            response = client.post(
                "/predict",
                files={"file": ("img.jpg", img, "image/jpeg")},
                data={"cancer_type": "brain"},
            )
        data = response.json()
        assert data["status"] == "error"
        assert data["reason"] == "model_unavailable"

    def test_cancer_type_case_insensitive(self):
        img  = _make_image_bytes()
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            response = client.post(
                "/predict",
                files={"file": ("img.jpg", img, "image/jpeg")},
                data={"cancer_type": "  Brain  "},
            )
        data = response.json()
        assert data["status"] == "success"

    def test_rejected_response_has_message(self):
        data = self._post(b"bad" * 400).json()
        assert "message" in data

    def test_oversized_file_returns_rejected(self):
        big  = b"x" * (21 * 1024 * 1024)
        data = self._post(big).json()
        assert data["status"] == "rejected"
        assert data["reason"] == "invalid_input"

    def test_non_medical_image_domain_screen_rejection(self):
        # Patch screen_image_domain to simulate a colourful (non-medical) image.
        # Verifies the HTTP layer correctly surfaces the "non_medical_image" reason.
        img  = _make_image_bytes()
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "non_medical_image", "saturation 0.45 exceeds brain limit 0.15"),
            ):
                response = client.post(
                    "/predict",
                    files={"file": ("img.jpg", img, "image/jpeg")},
                    data={"cancer_type": "brain"},
                )
        data = response.json()
        assert response.status_code == 200
        assert data["status"]  == "rejected"
        assert data["reason"]  == "non_medical_image"
        assert "message" in data

    def test_cross_modality_domain_screen_rejection(self):
        # Patch screen_image_domain to simulate histopathology sent to brain model.
        # Verifies the HTTP layer correctly surfaces the "cross_modality" reason.
        img  = _make_image_bytes()
        mods = _make_mock_models()
        with patch.dict(predict_module._models, mods):
            with patch(
                "backend.app.predict.screen_image_domain",
                return_value=(True, "cross_modality", "texture density 2500 exceeds brain maximum 1500"),
            ):
                response = client.post(
                    "/predict",
                    files={"file": ("img.jpg", img, "image/jpeg")},
                    data={"cancer_type": "brain"},
                )
        data = response.json()
        assert response.status_code == 200
        assert data["status"]  == "rejected"
        assert data["reason"]  == "cross_modality"
        assert "message" in data
