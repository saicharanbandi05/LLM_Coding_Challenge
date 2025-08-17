import numpy as np
from dataset_loader import load_fsdd_dataset
from feature_extraction import extract_features
from model_training import split_dataset, train_model, evaluate_model
from prediction import predict_digit
from recorder import record_and_predict_digit

def main():
    # 1. Load dataset
    ds = load_fsdd_dataset()

    # 2. Extract features
    print("Extracting features...")
    X = np.stack([extract_features(row) for row in ds])
    y = np.array([row['label'] for row in ds])

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # 4. Train model
    clf = train_model(X_train, y_train)

    # 5. Evaluate
    y_pred, acc = evaluate_model(clf, X_test, y_test)

    # 6. Example prediction
    idx = 0
    predicted = clf.predict([X_test[idx]])[0]
    true_label = y_test[idx]
    print(f"\nExample prediction: True label={true_label}, Predicted={predicted}")

    # try:
    #     while True:
    #         record_and_predict_digit(clf)
    # except KeyboardInterrupt:
    #     print("\nExiting live prediction.")
    
    predicted_digit = record_and_predict_digit(clf)
    print(f"\nPredicted digit from microphone: {predicted_digit}")

if __name__ == "__main__":
    main()
