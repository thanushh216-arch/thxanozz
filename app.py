from fastapi import FastAPI
import os
import requests
import numpy as np
import tensorflow as tf
from typing import List
MODEL_URL = "https://drive.google.com/uc?id=1e8aOjV6A9B1ZMakxfcdfSTrq9mj4gglh"
MODEL_PATH = "model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded")

download_model()

app = FastAPI()

# Load the model once when server starts
model = tf.keras.models.load_model("model.h5")

@app.get("/")
def home():
    return {"message": "ML server is running"}

@app.post("/predict")
def predict(data: list):
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}
