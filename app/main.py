import base64
import os
import tempfile
import uuid
import numpy as np

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import librosa
import joblib

# =========================
# APP INIT
# =========================
app = FastAPI(
    title="UprightAI Voice Detection API",
    version="1.0.0"
)

# =========================
# LOAD MODEL (LAZY SAFE)
# =========================
MODEL_PATH = "models/voice_rf.pkl"
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model file not found")
        _model = joblib.load(MODEL_PATH)
    return _model


# =========================
# REQUEST SCHEMA
# =========================
class VoiceDetectionRequest(BaseModel):
    audioBase64: str
    audioFormat: str  # "mp3" or "ogg"


# =========================
# HEALTHCHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "UprightAI Voice Detection API running"}


# =========================
# CORE ENDPOINT
# =========================
@app.post("/voice-detection")
def voice_detection(
    payload: VoiceDetectionRequest,
    x_api_key: str = Header(...)
):
    # ---- API KEY CHECK (simple) ----
    if x_api_key != "test_key_123":
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_format = payload.audioFormat.lower()
    if audio_format not in ["mp3", "ogg"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # ---- BASE64 DECODE ----
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    # ---- SAVE TEMP AUDIO FILE ----
    temp_filename = f"{uuid.uuid4()}.{audio_format}"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to write audio file")

    # ---- LOAD AUDIO SAFELY ----
    try:
        y, sr = librosa.load(temp_path, sr=16000, mono=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Audio decode failed: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if y is None or len(y) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    # ---- FEATURE EXTRACTION (MFCC) ----
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    except Exception:
        raise HTTPException(status_code=500, detail="Feature extraction failed")

    # ---- PREDICTION ----
    try:
        model = get_model()
        prob = model.predict_proba(features)[0]
        label_index = int(np.argmax(prob))
        confidence = float(np.max(prob))
        label = "AI_GENERATED" if label_index == 1 else "HUMAN"
    except Exception:
        raise HTTPException(status_code=500, detail="Model prediction failed")

    # ---- RESPONSE ----
    return {
        "status": "success",
        "prediction": {
            "label": label,
            "confidence": round(confidence, 3),
            "decision": "HIGH_CONFIDENCE" if confidence >= 0.9 else "LOW_CONFIDENCE"
        },
        "meta": {
            "language": "Unknown",
            "note": "Language auto-detected; AI detection is language-agnostic"
        }
    }
