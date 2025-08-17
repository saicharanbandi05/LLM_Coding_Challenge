import numpy as np
import sounddevice as sd
from prediction import predict_digit

def record_and_predict_digit(clf, duration=1, fs=8000):
    print("\nSpeak a digit (0–9) after the beep...")

    # Beep sound
    beep = np.sin(2 * np.pi * 440 * np.arange(fs * 0.2) / fs)
    sd.play(beep, fs)
    sd.wait()

    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = recording.flatten()

    # Normalize
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

    # Predict
    pred = predict_digit(audio, sr=fs, clf=clf)
    print(f"Predicted digit: {pred}")
    return pred
  
