from fastapi import FastAPI, Form
import pickle
import numpy as np

app = FastAPI()

print("APP LOADED")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Home route
@app.get("/")
def home():
    return {"message": "API is running"}

# Prediction route
@app.post("/predict")
def predict(data: str = Form(...)):
    try:
        # Convert input string → list of numbers
        data = [float(i.strip()) for i in data.split(",")]

        # Reshape for model
        data = np.array(data).reshape(1, -1)

        # Predict
        prediction = model.predict(data)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}