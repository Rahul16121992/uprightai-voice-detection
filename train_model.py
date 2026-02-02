import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from app.features import extract_features

# -----------------------------
# Data containers
# -----------------------------
X = []
y = []

# -----------------------------
# HUMAN VOICE DATA  -> label 0
# Folder: data/human_voice
# -----------------------------
human_folder = "data/human_voice"

for file in os.listdir(human_folder):
    if file.endswith(".mp3"):
        path = os.path.join(human_folder, file)
        audio, sr = librosa.load(path, sr=None)
        features = extract_features(audio, sr)
        X.append(features)
        y.append(0)

# -----------------------------
# AI GENERATED VOICE -> label 1
# Folder: data/ai_voice
# -----------------------------
ai_folder = "data/ai_voice"

for file in os.listdir(ai_folder):
    if file.endswith(".mp3"):
        path = os.path.join(ai_folder, file)
        audio, sr = librosa.load(path, sr=None)
        features = extract_features(audio, sr)
        X.append(features)
        y.append(1)

# -----------------------------
# Convert to numpy arrays
# -----------------------------
X = np.array(X)
y = np.array(y)

print("Training samples shape:", X.shape)
print("Labels count:", np.bincount(y))

# -----------------------------
# Train RandomForest model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# -----------------------------
# Save trained model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/voice_rf.pkl")

print("✅ RandomForest model trained & saved at models/voice_rf.pkl")
