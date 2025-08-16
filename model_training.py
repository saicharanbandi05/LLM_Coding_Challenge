import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into train and test subsets.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_model(X_train, y_train):
    """
    Trains a Random Forest model on training data.
    """
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """
    Evaluates trained classifier and prints report.
    """
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy: {:.2f}%".format(acc*100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return y_pred, acc
