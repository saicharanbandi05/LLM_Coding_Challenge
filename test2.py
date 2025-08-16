# import sounddevice as sd

# def record_and_predict_digit(duration=1, fs=8000):
#     """
#     Records audio from your microphone and predicts spoken digit in real time.
#     """
#     print("\nSpeak a digit (0–9) after the beep...")
#     # Play a quick 'beep'
#     sd.play(np.sin(2*np.pi*440*np.arange(fs*0.2)/fs), fs)
#     sd.wait()
#     # Record audio for specified duration
#     recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
#     sd.wait()
#     audio = recording.flatten()
#     # Normalize
#     audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
#     # Predict
#     pred = predict_digit(audio, sr=fs)
#     print(f"Predicted digit: {pred}")

# # Uncomment this to test live microphone prediction
# record_and_predict_digit()
