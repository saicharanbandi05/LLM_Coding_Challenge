import numpy as np
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the FSDD dataset from Hugging Face
print("Loading dataset...")
ds = load_dataset("mteb/free-spoken-digit-dataset", split="train")

# 2. Feature Extraction - extract MFCCs from each audio sample
def extract_features(entry, n_mfcc=13, max_len=8000):
    # 'audio' field contains a dictionary: {'path', 'array', 'sampling_rate'}
    arr = entry['audio']['array']
    sr = entry['audio']['sampling_rate']
    # Pad or trim to max_len samples (1s at 8kHz)
    if len(arr) > max_len:
        arr = arr[:max_len]
    elif len(arr) < max_len:
        arr = np.pad(arr, (0, max_len - len(arr)))
    # Compute MFCCs (shape: [n_mfcc, time])
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # [n_mfcc] vector

# Extract MFCC feature vectors and labels
print("Extracting features...")
X = np.stack([extract_features(row) for row in ds])
y = np.array([row['label'] for row in ds])

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Train a lightweight classifier
print("Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# 5. Evaluate model
y_pred = clf.predict(X_test)
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Predict from new audio (example: reusing a test sample)
def predict_digit(audio_arr, sr):
    arr = audio_arr
    # Pad/truncate as above
    if len(arr) > 8000:
        arr = arr[:8000]
    elif len(arr) < 8000:
        arr = np.pad(arr, (0, 8000 - len(arr)))
    mfcc = librosa.feature.mfcc(y=arr, sr=sr, n_mfcc=13)
    feat = np.mean(mfcc, axis=1)
    digit = clf.predict([feat])[0]
    return digit

# 7. Test single prediction on test set features directly
idx = 0  # Can choose any valid index for the test set
predicted = clf.predict([X_test[idx]])[0]
true_label = y_test[idx]
print(f"\nExample prediction: True label={true_label}, Predicted={predicted}")
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred)*100))