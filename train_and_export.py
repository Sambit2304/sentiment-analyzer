from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from backend.text_preprocess import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

TRAIN_CSV = BASE_DIR / "twitter_training.csv"
VALID_CSV = BASE_DIR / "twitter_validation.csv"


def load_and_clean_training(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=["TweetID", "Topic", "Sentiment", "Text"])

    # Same cleaning pipeline as the notebook.
    valid_sentiments = ["Positive", "Negative", "Neutral", "Irrelevant"]

    df_clean = df.dropna(subset=["Text"]).copy()
    df_clean = df_clean[df_clean["Text"].astype(str).str.strip().str.len() >= 3]

    df_clean["Text"] = df_clean["Text"].astype(str).str.strip()
    df_clean["Sentiment"] = df_clean["Sentiment"].astype(str).str.strip()
    df_clean["Topic"] = df_clean["Topic"].astype(str).str.strip()

    df_clean = df_clean.drop_duplicates(subset=["Text"], keep="first")

    invalid_mask = ~df_clean["Sentiment"].isin(valid_sentiments)
    df_clean = df_clean[~invalid_mask]

    # Apply text normalization for model features.
    df_clean["CleanedText"] = df_clean["Text"].apply(clean_text)
    df_clean = df_clean[df_clean["CleanedText"].str.len() >= 3]
    df_clean["CleanedWordCount"] = df_clean["CleanedText"].apply(lambda x: len(x.split()))

    return df_clean


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing training CSV: {TRAIN_CSV}")

    # Load + clean
    df_clean = load_and_clean_training(TRAIN_CSV)

    X_train = df_clean["CleanedText"].values
    y_train = df_clean["Sentiment"].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    # TF-IDF config from the notebook
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = tfidf.fit_transform(X_train)

    # Fast deployment-friendly model: Linear SVM
    model = LinearSVC(max_iter=2000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train_enc)

    # Export artifacts
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.joblib"
    model_path = MODELS_DIR / "sentiment_model.joblib"
    label_enc_path = MODELS_DIR / "label_encoder.joblib"
    metadata_path = MODELS_DIR / "metadata.json"

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(model, model_path)
    joblib.dump(le, label_enc_path)

    metadata = {
        "model_name": "LinearSVC",
        "label_classes": list(le.classes_),
        "vectorizer_max_features": getattr(tfidf, "max_features", None),
        "ngram_range": getattr(tfidf, "ngram_range", None),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model artifacts exported successfully to `models/`:")
    print(f"  {tfidf_path}")
    print(f"  {model_path}")
    print(f"  {label_enc_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()

