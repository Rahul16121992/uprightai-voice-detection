# app/features.py
import numpy as np
import librosa

def extract_features(y, sr):
    """
    TOTAL FEATURES = 13
    SAME features training + inference
    """

    # MFCC (13 coefficients → mean)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean
