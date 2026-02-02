from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import io
import numpy as np
import librosa
import speech_recognition as sr

from app.features import extract_features
from app.model import predict_rf

app = FastAPI(
    title="UprightAI Voice Detection API",
    version="1.0.0"
)

# -----------------------------
# Request schema
# -----------------------------
class VoiceRequest(BaseModel):
    audioFormat: str
    audioBase64: str


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "API is running"}


# -----------------------------
# Main API (GUVI compliant)
# -----------------------------
@app.post("/voice-detection")
def voice_detection(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != "test_key_123":
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(data.audioBase64)
        audio_buffer = io.BytesIO(audio_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Load audio
    try:
        y, sr_audio = librosa.load(audio_buffer, sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read audio")

    # Feature extraction
    features = extract_features(y, sr_audio)

    # RandomForest prediction
    pred, probs = predict_rf(features)

    if pred == 0:
        classification = "HUMAN"
        confidence = round(float(probs[0]), 2)
    else:
        classification = "AI_GENERATED"
        confidence = round(float(probs[1]), 2)

    # -----------------------------
    # SAFE LANGUAGE HANDLING
    # -----------------------------
    # We do NOT guess language.
    # Only English or Unknown is returned.
    detected_language = "English"

    try:
        recognizer = sr.Recognizer()
        audio_buffer.seek(0)
        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source, duration=4)
            recognizer.recognize_google(audio_data)
            detected_language = "English"
    except Exception:
        detected_language = "Unknown"

    # -----------------------------
    # FINAL RESPONSE (GUVI FORMAT)
    # -----------------------------
    return {
        "status": "success",
        "language": detected_language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": "RandomForest-based acoustic pattern analysis using MFCC and spectral features"
    }
