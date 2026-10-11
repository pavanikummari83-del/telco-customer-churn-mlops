
import json
import joblib
import numpy as np


def test_actual_model_predictions_are_valid():
    model = joblib.load("churn_model.joblib")
    X_test = joblib.load("X_test.joblib")

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_actual_model_accuracy_and_f1():
    with open("metrics.json") as file:
        metrics = json.load(file)

    assert metrics["accuracy"] >= 0.70
    assert metrics["f1_score"] >= 0.45


def test_model_can_predict_one_customer():
    model = joblib.load("churn_model.joblib")
    X_test = joblib.load("X_test.joblib")

    prediction = model.predict(X_test.iloc[:1])

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)
