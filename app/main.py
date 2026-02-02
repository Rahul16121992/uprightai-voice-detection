from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import librosa
import numpy as np
import joblib
import os

app = FastAPI(title="UprightAI Voice Detection API")

# -------------------------
# Healthcheck (VERY IMPORTANT)
# -------------------------
@app.get("/")
def healthcheck():
    return {"status": "ok"}

# -------------------------
# Request Schema
# -------------------------
class VoiceRequest(BaseModel):
    audioBase64: str

# -------------------------
# Lazy Model Loader (CRITICAL FIX)
# -------------------------
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load("models/voice_rf.pkl")
    return model

# -------------------------
# Feature Extraction (MFCC)
# -------------------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# -------------------------
# Main API Endpoint
# -------------------------
@app.post("/voice-detection")
def voice_detection(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key check
    if x_api_key != "test_key_123":
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio data")

    # Feature extraction
    features = extract_features(y, sr)

    # Prediction
    rf_model = get_model()
    pred = rf_model.predict(features)[0]
    probs = rf_model.predict_proba(features)[0]
    confidence = float(np.max(probs))

    label = "AI_GENERATED" if pred == 1 else "HUMAN"

    return {
        "status": "success",
        "prediction": {
            "label": label,
            "confidence": round(confidence, 2),
            "decision": "HIGH_CONFIDENCE" if confidence >= 0.8 else "LOW_CONFIDENCE"
        },
        "model": {
            "type": "RandomForestClassifier",
            "features": ["MFCC"],
            "version": "v1.0"
        },
        "meta": {
            "language": "Unknown",
            "note": "Language auto-detection not required; AI detection is language-agnostic"
        }
    }
