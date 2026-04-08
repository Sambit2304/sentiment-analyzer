import logging
import os
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

from .models_loader import load_artifacts, predict_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter Sentiment API", version="1.0.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Load model artifacts once at startup.
    # If artifacts are missing, we keep the server running and fail
    # individual predictions with a clear message.
    try:
        load_artifacts()
    except RuntimeError as e:
        logger.warning("%s", str(e))


class SentimentRequest(BaseModel):
    text: str = Field(..., description="User input text")


class SentimentResponse(BaseModel):
    sentiment: str
    sentiment_code: int
    probabilities: Optional[Dict[str, float]] = None
    cleaned_text: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest) -> SentimentResponse:
    try:
        # Enforce stricter validation than Field(min_length=1) since we also strip
        text = req.text
        if text is None or not str(text).strip():
            raise ValueError("Text must not be empty.")

        result = predict_sentiment(text)
        return SentimentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")


# Serve the frontend (production-friendly).
# This makes the API and UI run behind the same URL (recommended for deploys).
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")

