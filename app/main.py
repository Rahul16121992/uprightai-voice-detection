from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os

from app.features import extract_features
from app.model import predict_rf

# ======================
# FastAPI App
# ======================
app = FastAPI(
    title="UprightAI Voice Detection API",
    version="1.1.0",
    description="AI Generated vs Human Voice Detection API"
)

# ======================
# HEALTHCHECK (VERY IMPORTANT FOR RAILWAY)
# ======================
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# ======================
# Request Schema
# ======================
class VoiceDetectionRequest(BaseModel):
    audioBase64: str

# ======================
# Main API Endpoint
# ======================
@app.post("/voice-detection")
def voice_detection(
    payload: VoiceDetectionRequest,
    x_api_key: str = Header(None)
):
    # ---- API Key Check ----
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key")

    # ---- Decode Base64 Audio ----
    try:
        audio_bytes = base64.b64decode(payload.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # ---- Save temp audio file ----
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        # ---- Feature Extraction ----
        features = extract_features(audio_path)

        # ---- ML Prediction ----
        label, confidence = predict_rf(features)

    finally:
        # ---- Cleanup ----
        if os.path.exists(audio_path):
            os.remove(audio_path)

    # ---- Response ----
    return {
        "status": "success",
        "prediction": {
            "label": label,
            "confidence": round(float(confidence), 3),
            "decision": "HIGH_CONFIDENCE" if confidence >= 0.9 else "LOW_CONFIDENCE"
        },
        "model": {
            "type": "RandomForestClassifier",
            "features": ["MFCC", "Spectral"],
            "version": "v1.1"
        },
        "meta": {
            "language": "Unknown",
            "note": "Language auto-detected; AI detection is language-agnostic"
        }
    }
