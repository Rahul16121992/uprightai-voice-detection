# train_model.py
import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from app.features import extract_features

DATA_DIR = "data"
AI_DIR = os.path.join(DATA_DIR, "ai_voice")
HUMAN_DIR = os.path.join(DATA_DIR, "human_voice")
MODEL_PATH = "models/voice_rf.pkl"

X = []
y = []

def load_folder(folder_path, label):
    for file in os.listdir(folder_path):
        if file.endswith(".mp3") or file.endswith(".wav") or file.endswith(".ogg"):
            path = os.path.join(folder_path, file)
            try:
                audio, sr = librosa.load(path, sr=None)
                features = extract_features(audio, sr)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Skipping {file}: {e}")

print("📂 Loading AI voices...")
load_folder(AI_DIR, 1)

print("📂 Loading Human voices...")
load_folder(HUMAN_DIR, 0)

X = np.array(X)
y = np.array(y)

print("Training samples shape:", X.shape)
print("Labels count:", np.bincount(y))

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model trained & saved at {MODEL_PATH}")
