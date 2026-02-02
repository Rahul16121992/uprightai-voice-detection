from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import os
import numpy as np
import librosa
import joblib

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="UprightAI Voice Detection API",
    version="1.0.0"
)

# -----------------------------
# Request schema
# -----------------------------
class VoiceRequest(BaseModel):
    audioBase64: str
    audioFormat: str = "mp3"

# -----------------------------
# Lazy-loaded model
# -----------------------------
MODEL = None
MODEL_PATH = "models/voice_rf.pkl"

def load_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found")
        MODEL = joblib.load(MODEL_PATH)

# -----------------------------
# Healthcheck (Railway needs this)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "service": "AI Generated Voice Detection API",
        "status": "running"
    }

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([
        mfcc_mean,
        spectral_centroid,
        spectral_bandwidth,
        zero_crossing
    ])

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/voice-detection")
def voice_detection(
    payload: VoiceRequest,
    x_api_key: str = Header(None)
):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    try:
        # Load model lazily
        load_model()

        # Decode audio
        audio_bytes = base64.b64decode(payload.audioBase64)
        audio_buffer = io.BytesIO(audio_bytes)

        y, sr = librosa.load(audio_buffer, sr=None)

        features = extract_features(y, sr).reshape(1, -1)

        probs = MODEL.predict_proba(features)[0]
        pred = MODEL.predict(features)[0]

        label = "AI_GENERATED" if pred == 1 else "HUMAN"
        confidence = float(np.max(probs))

        decision = (
            "HIGH_CONFIDENCE" if confidence >= 0.8
            else "MEDIUM_CONFIDENCE" if confidence >= 0.6
            else "LOW_CONFIDENCE"
        )

        return {
            "status": "success",
            "prediction": {
                "label": label,
                "confidence": round(confidence, 4),
                "decision": decision
            },
            "model": {
                "type": "RandomForestClassifier",
                "features": ["MFCC", "Spectral"],
                "version": "v1.1"
            },
            "meta": {
                "language": "unknown",
                "note": "Language auto-detected; AI detection is language-agnostic"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
