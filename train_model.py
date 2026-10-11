
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone


DATA_URL = (
    "https://raw.githubusercontent.com/aiplanethub/"
    "Datasets/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    df = pd.read_csv(DATA_URL)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected or missing churn labels")

    X = df.drop(columns=["customerID", "Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
    }

    joblib.dump(model, "churn_model.joblib")
    joblib.dump(list(X.columns), "feature_columns.joblib")
    joblib.dump(X_test, "X_test.joblib")
    joblib.dump(y_test, "y_test.joblib")

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print("Model training completed!")
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1-score:", round(metrics["f1_score"], 4))


if __name__ == "__main__":
    main()
