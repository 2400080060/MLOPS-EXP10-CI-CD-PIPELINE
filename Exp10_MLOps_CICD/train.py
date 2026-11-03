"""
Train a simple Iris classifier and save the trained model.
Run locally in PyCharm: python train.py
"""
import os
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

MODEL_VERSION = os.getenv("MODEL_VERSION", "local-1.0")

def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42,
        stratify=iris.target
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/iris_model.joblib")

    print("=== MLOps Model Training ===")
    print(f"Model version : {MODEL_VERSION}")
    print(f"Training rows : {len(X_train)}")
    print(f"Testing rows  : {len(X_test)}")
    print(f"Accuracy      : {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names
    ))
    print("Model saved   : artifacts/iris_model.joblib")
    return model, accuracy

if __name__ == "__main__":
    train_model()
