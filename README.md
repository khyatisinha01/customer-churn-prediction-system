# Customer Churn Prediction System

An end-to-end machine learning pipeline that predicts customer churn using the IBM Telco Customer Churn dataset. Includes data preprocessing, model benchmarking, and a production-ready REST API deployed with FastAPI and Docker.

---

## Results

| Model | CV Accuracy (5-fold) | Test Accuracy | F1 Score | Precision | Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | **80.21% ± 1.15%** | **80.38%** | **0.6091** | **0.6476** | **0.5749** |
| Random Forest | 79.32% ± 0.56% | 78.96% | 0.5673 | 0.6258 | 0.5187 |
| Decision Tree | 73.51% ± 0.81% | 71.86% | 0.4663 | 0.4701 | 0.4626 |

**Best model:** Logistic Regression, selected based on F1 score (accounts for class imbalance — 26.6% churn rate).

---

## Project Structure

```
├── churn.csv           # IBM Telco Customer Churn dataset (7,032 records)
├── train.py            # Data preprocessing, benchmarking, model training
├── app.py              # FastAPI REST API for real-time inference
├── model.pkl           # Serialized trained model
├── requirements.txt    # Dependencies
└── Dockerfile          # Container setup
```

---

## Tech Stack

- **Python**, **Scikit-learn**, **Pandas**, **NumPy**
- **FastAPI** — REST API with Pydantic input validation
- **Docker** — containerized deployment
- **Pickle** — model serialization

---

## Run Locally

### Option 1 — Python
```bash
pip install -r requirements.txt
python train.py          # Train and evaluate model
uvicorn app:app --reload # Start API at http://localhost:8000
```

### Option 2 — Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## API Usage

**Endpoint:** `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 12, 1, 0, 1, 0, 1, 0, 29.85, 29.85, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]}'
```

**Response:**
```json
{
  "prediction": 0,
  "label": "No Churn",
  "churn_probability": 0.2341
}
```

Interactive docs available at `http://localhost:8000/docs`

---

## Key Design Decisions

- Used **F1 score** (not accuracy) as the primary selection metric due to class imbalance (26.6% churn)
- Applied **5-fold stratified cross-validation** to avoid evaluation bias from a single train-test split
- Used **StandardScaler** for Logistic Regression; tree-based models trained on unscaled features
- **Pydantic** input validation in the API replaces raw string parsing for robustness
