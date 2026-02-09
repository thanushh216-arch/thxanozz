import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from typing import List

# Create FastAPI app
app = FastAPI()

# Model path (LOCAL file inside Docker/GitHub)
MODEL_PATH = "model.keras"

# Load model safely
print("✅ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("🚀 Model loaded successfully")

@app.post("/predict")
def predict(data: List[List[List[List[float]]]]):
    """
    Expected shape: (1, 128, 128, 1)
    """
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}
