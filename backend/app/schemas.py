"""
Pydantic response schemas for the prediction API.
All fields are optional so a single model can represent every possible
response shape: success, rejected, or error.
"""

from typing import Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    # Always present
    status: str                              # "success" | "rejected" | "error"

    # Present on success
    cancer_type    : Optional[str] = None
    predicted_class: Optional[str] = None

    # Present on rejection / error
    reason : Optional[str] = None   # "invalid_input"|"invalid_cancer_type"|"model_unavailable"
    message: Optional[str] = None   # human-readable explanation


class HealthResponse(BaseModel):
    status        : str
    models_loaded : list[str]
    models_missing: list[str]
