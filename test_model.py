
import numpy as np
from sklearn.dummy import DummyClassifier

def test_model_predictions_are_valid():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1})
