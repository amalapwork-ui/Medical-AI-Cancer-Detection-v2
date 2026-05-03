"""
FastAPI application — Medical AI Cancer Detection v2
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_CONTENT_TYPES
from .predict import load_models, loaded_models, missing_models
from .predict import predict as run_predict
from .schemas import HealthResponse, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan: load models once at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[startup] Server starting — loading models in background thread …")
    # load_models() calls tf.keras.models.load_model() which is synchronous and
    # CPU/GPU-heavy.  Running it directly in the async lifespan would block the
    # event loop for the full load time (15–60 s for 4 EfficientNetV2S models).
    # asyncio.to_thread() offloads it to the default ThreadPoolExecutor so the
    # event loop stays responsive while models are loading.
    await asyncio.to_thread(load_models)
    logger.info("[startup] Ready.  Loaded: %s  Missing: %s", loaded_models(), missing_models())
    yield
    logger.info("[shutdown] Server shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Medical AI Cancer Detection API",
    description = "Multi-organ cancer classification with confidence-based quality gate.",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Reports which models are loaded and which are missing."""
    return HealthResponse(
        status        = "ok" if not missing_models() else "degraded",
        models_loaded = loaded_models(),
        models_missing= missing_models(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict_endpoint(
    file       : UploadFile = File(...,  description="Medical image (JPEG / PNG / BMP / WebP)"),
    cancer_type: str        = Form(...,  description="One of: brain | lung | colon"),
):
    """
    Full inference pipeline:
    1. Validate uploaded file format and size.
    2. Classify cancer type and apply confidence quality gate.
    3. Return probability distribution and prediction status.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code = 415,
            detail      = (
                f"Unsupported content type '{file.content_type}'. "
                f"Accepted: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    logger.info("[endpoint /predict] file=%r  type=%r  cancer=%r",
                file.filename, file.content_type, cancer_type)

    image_bytes = await file.read()

    # run_predict() is synchronous (PIL decode + TF inference).
    # Calling it directly from an async handler would block the event loop for
    # the full inference duration, freezing ALL concurrent requests on Windows.
    # asyncio.to_thread() runs it in a thread so the event loop stays free.
    result = await asyncio.to_thread(run_predict, image_bytes, cancer_type)

    logger.info("[endpoint /predict] Response status=%r", result.get("status"))
    return result
