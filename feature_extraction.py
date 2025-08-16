import numpy as np
import librosa

def extract_features(entry, n_mfcc=13, max_len=8000):
    """
    Extract MFCC features from an audio sample.
    Pads or trims to fixed length.
    """
    arr = entry['audio']['array']
    sr = entry['audio']['sampling_rate']

    # Pad or trim
    if len(arr) > max_len:
        arr = arr[:max_len]
    elif len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)))

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # feature vector
