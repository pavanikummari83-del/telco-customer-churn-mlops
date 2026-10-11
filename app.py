
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Telco Customer Churn Prediction API")

MODEL_PATH = "churn_model.joblib"
FEATURE_PATH = "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


class Customer(BaseModel):
    data: dict


@app.get("/")
def home():
    return {
        "message": "Telco Customer Churn API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: Customer):
    try:
        input_df = pd.DataFrame([customer.data])
        input_df = input_df.reindex(columns=feature_columns)

        prediction = int(model.predict(input_df)[0])

        return {
            "prediction": prediction,
            "result": (
                "Customer may leave"
                if prediction == 1
                else "Customer may stay"
            )
        }
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid customer data or prediction input."
        ) from error
