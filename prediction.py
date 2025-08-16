import numpy as np
import librosa

def predict_digit(audio_arr, sr, clf, max_len=8000, n_mfcc=13):
    arr = audio_arr
    if len(arr) > max_len:
        arr = arr[:max_len]
    elif len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)))
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=n_mfcc)
    feat = np.mean(mfcc, axis=1)
    return clf.predict([feat])[0]