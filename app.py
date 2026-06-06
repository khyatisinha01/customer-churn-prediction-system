from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn using a trained Logistic Regression model.",
    version="1.0.0"
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


class ChurnInput(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0, 12, 1, 0, 1, 0, 1, 0, 29.85, 29.85,
                              1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                              1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
            }
        }


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running", "docs": "/docs"}


@app.post("/predict")
def predict(input: ChurnInput):
    try:
        data = np.array(input.features).reshape(1, -1)
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]
        return {
            "prediction": int(prediction[0]),
            "label": "Churn" if prediction[0] == 1 else "No Churn",
            "churn_probability": round(float(probability), 4)
        }
    except Exception as e:
        return {"error": str(e)}
