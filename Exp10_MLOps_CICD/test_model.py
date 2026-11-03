"""Basic automated tests for the Iris ML model. Run: pytest -v"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def build_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42,
        stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def test_model_trains_and_predicts():
    model, X_test, y_test = build_model()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_model_accuracy_is_acceptable():
    model, X_test, y_test = build_model()
    assert model.score(X_test, y_test) >= 0.85
