# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define input data structure
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# Root endpoint to redirect to /docs
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")


# Prediction endpoint
@app.post("/predict")
def predict_quality(features: WineFeatures):
    data = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
    # Rename columns to match the model's expected feature names
    data.columns = [col.replace('_', ' ') for col in data.columns]  # Change underscores to spaces
    prediction = model.predict(data)[0]
    return {"quality": "Good" if prediction == 1 else "Bad"}