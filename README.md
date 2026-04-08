# Twitter Sentiment Analyzer (FastAPI + Modern Web UI)

A production-ready sentiment analysis web app.
- Backend: FastAPI (loads a saved scikit-learn model on startup)
- Frontend: Tailwind CSS (via CDN) + vanilla JavaScript
- Model: TF-IDF + classifier + label encoder exported to `models/` and loaded at runtime

The backend also serves the frontend UI (so you can deploy as a single Docker service).

## Live / Deployed

After deploying (see **Deploy to Render**), open the service URL in your browser.

## Features

- Smooth, centered text input UI
- Loading spinner while the API runs
- Result badge with sentiment-aware styling
- `/health` endpoint for uptime checks

## Tech Stack

- Python 3.13
- FastAPI
- scikit-learn
- joblib
- Tailwind CSS (CDN)
- vanilla JavaScript
- Docker

## Project Structure

- `backend/`
  - `main.py`: FastAPI app + static frontend mounting + routes
  - `models_loader.py`: loads `models/*.joblib` and implements `predict_sentiment`
  - `text_preprocess.py`: preprocessing used during inference (matches notebook)
- `frontend/`
  - `index.html`: single page UI
  - `app.js`: fetches `/predict` and animates output
- `models/`
  - `tfidf_vectorizer.joblib`
  - `sentiment_model.joblib`
  - `label_encoder.joblib`
  - `metadata.json`
- `train_and_export.py`: trains a deploy-ready model and writes the artifacts to `models/`
- `Dockerfile`: container entrypoint for FastAPI
- `requirements.txt`: Python dependencies

## Prerequisites

- Python (to run locally)
- Docker (recommended for deployment)
- GitHub repo (for Render deployment)

## Setup (Local Development)

From the project root (`.`):

### 1) Create and activate a virtual environment

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and Export Model Artifacts

If you need to regenerate model artifacts (or if `models/` is missing):

```powershell
.\.venv\Scripts\python.exe train_and_export.py
```

This creates:
- `models/tfidf_vectorizer.joblib`
- `models/sentiment_model.joblib`
- `models/label_encoder.joblib`

## Run the Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Frontend: `http://localhost:8000/`

## Run the Frontend (Optional)

Because the backend serves the frontend in production, this is optional for local testing.

If you want to serve static files separately:

```powershell
cd frontend
python -m http.server 5173
```

Open: `http://localhost:5173`

## API Reference

### `GET /health`

Response:

```json
{ "status": "ok" }
```

### `POST /predict`

Request:

```json
{ "text": "Your input text here" }
```

Response:

```json
{
  "sentiment": "Positive",
  "sentiment_code": 3,
  "probabilities": {
    "Positive": 0.86,
    "Neutral": 0.04,
    "Negative": 0.05,
    "Irrelevant": 0.05
  },
  "cleaned_text": "..."
}
```

## Deploy to Render (Online)

This deployment uses **one Docker service** that runs FastAPI and serves the frontend UI.

### Steps

1. Push the repository to GitHub (including the `models/` folder and `Dockerfile`).
2. Create a Render:
   - **New +** -> **Web Service**
   - Choose **Docker**
   - Connect your GitHub repository
   - Root directory: `/` (repo root)
   - Dockerfile path: `/Dockerfile`
3. Configure health check:
   - Path: `/health`
   - Protocol: HTTP
4. Deploy and wait for Render to build and start the container.

After deployment:
- Frontend UI: `https://<your-service-url>/`
- API docs: `https://<your-service-url>/docs`
- Predict: `https://<your-service-url>/predict`

## Notes

- `models/` is required for inference. The repo intentionally does **not** ignore `models/*.joblib`.
- If you change preprocessing logic, you must regenerate the artifacts using `train_and_export.py`.

## License

MIT

