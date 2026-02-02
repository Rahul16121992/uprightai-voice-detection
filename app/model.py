import joblib
import numpy as np

model = joblib.load("models/voice_rf.pkl")

def predict_rf(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return prediction, probabilities
