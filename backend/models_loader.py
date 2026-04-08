import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from .text_preprocess import clean_text

logger = logging.getLogger(__name__)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_artifacts() -> tuple[Any, Any, Any]:
    """
    Load TF-IDF vectorizer, classifier, and label encoder from `models/`.

    Expected filenames:
      - tfidf_vectorizer.joblib
      - sentiment_model.joblib
      - label_encoder.joblib
    """
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    model_path = MODELS_DIR / "sentiment_model.joblib"
    le_path = MODELS_DIR / "label_encoder.joblib"

    missing = [str(p) for p in [vec_path, model_path, le_path] if not p.exists()]
    if missing:
        raise RuntimeError(
            "Missing model artifacts in `models/`. "
            f"Expected: tfidf_vectorizer.joblib, sentiment_model.joblib, label_encoder.joblib. "
            f"Missing: {missing}"
        )

    logger.info("Loading model artifacts from %s", MODELS_DIR)
    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    return vectorizer, model, label_encoder


VECTORIZER = None
MODEL = None
LABEL_ENCODER = None
_ARTIFACTS_LOADED = False


def load_artifacts() -> None:
    """
    Load TF-IDF vectorizer, classifier, and label encoder from `models/`.

    This is intentionally lazy (called from FastAPI startup) so importing the
    module doesn't crash before the artifacts are present.
    """
    global VECTORIZER, MODEL, LABEL_ENCODER, _ARTIFACTS_LOADED
    if _ARTIFACTS_LOADED:
        return

    VECTORIZER, MODEL, LABEL_ENCODER = _load_artifacts()
    _ARTIFACTS_LOADED = True


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment for a single text input.

    Returns:
      {
        "sentiment": <label string>,
        "sentiment_code": <int encoded label>,
        "probabilities": {<label>: <float>, ...} | None,
        "cleaned_text": <cleaned string>
      }
    """
    if text is None:
        raise ValueError("Text is required.")

    raw = str(text).strip()
    if not raw:
        raise ValueError("Text must not be empty.")

    cleaned = clean_text(raw)
    if not cleaned:
        raise ValueError("Text is empty after cleaning.")

    if not _ARTIFACTS_LOADED:
        load_artifacts()

    # Vectorizer expects an iterable of texts
    X = VECTORIZER.transform([cleaned])

    # Classifier output is encoded labels (LabelEncoder was fit on sentiments)
    pred_enc = MODEL.predict(X)
    if len(pred_enc) < 1:
        raise RuntimeError("Model returned no predictions.")

    pred_enc_val = int(pred_enc[0])

    # Decode encoded integer back to sentiment label string
    if hasattr(LABEL_ENCODER, "inverse_transform"):
        sentiment = str(LABEL_ENCODER.inverse_transform([pred_enc_val])[0])
    else:
        # Fallback for unexpected LabelEncoder-like objects
        classes = getattr(LABEL_ENCODER, "classes_", None) or []
        sentiment = str(classes[pred_enc_val]) if pred_enc_val < len(classes) else "Neutral"

    probabilities: Optional[Dict[str, float]] = None

    # Try to return probability-like values for a nicer UI.
    # Not all sklearn estimators support predict_proba.
    try:
        proba_map: Optional[Dict[str, float]] = None

        if hasattr(MODEL, "predict_proba"):
            proba_arr = MODEL.predict_proba(X)[0]  # shape: (n_classes,)

            # Use MODEL.classes_ to align with output order if present
            if hasattr(MODEL, "classes_"):
                enc_labels = [int(x) for x in MODEL.classes_]
            else:
                enc_labels = list(range(len(proba_arr)))

            proba_map = {}
            for enc, p in zip(enc_labels, proba_arr):
                if hasattr(LABEL_ENCODER, "inverse_transform"):
                    label_str = str(LABEL_ENCODER.inverse_transform([int(enc)])[0])
                else:
                    label_str = str(enc)
                proba_map[label_str] = float(p)

        elif hasattr(MODEL, "decision_function"):
            scores = MODEL.decision_function(X)[0]  # shape: (n_classes,)
            proba_arr = _softmax(scores)

            if hasattr(MODEL, "classes_"):
                enc_labels = [int(x) for x in MODEL.classes_]
            else:
                enc_labels = list(range(len(proba_arr)))

            proba_map = {}
            for enc, p in zip(enc_labels, proba_arr):
                if hasattr(LABEL_ENCODER, "inverse_transform"):
                    label_str = str(LABEL_ENCODER.inverse_transform([int(enc)])[0])
                else:
                    label_str = str(enc)
                proba_map[label_str] = float(p)

        probabilities = proba_map
    except Exception:
        # Probabilities are optional; don't fail prediction if not available.
        probabilities = None

    return {
        "sentiment": sentiment,
        "sentiment_code": pred_enc_val,
        "probabilities": probabilities,
        "cleaned_text": cleaned,
    }

