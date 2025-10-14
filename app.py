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
    fixed_acidity: float=8.5
    volatile_acidity: float=0.28
    citric_acid: float=0.56
    residual_sugar: float=1.8
    chlorides: float=0.092
    free_sulfur_dioxide: float=35
    total_sulfur_dioxide: float=103
    density: float=0.9969
    pH: float=3.16
    sulphates: float=0.75
    alcohol: float=10.5


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

@app.get("/hello")
def print_hello():
    return {"msg": "Hello World"}