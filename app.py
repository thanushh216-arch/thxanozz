import os
import requests
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from typing import List

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?id=1e8aOjV6A9B1ZMakxfcdfSTrq9mj4gglh"
MODEL_PATH = "model.pkl"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded")

# ✅ STEP 1: Download model
download_model()

# ✅ STEP 2: Ensure file exists before loading
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file was not downloaded successfully")

# ✅ STEP 3: Load model
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "ML server is running"}

@app.post("/predict")
def predict(data: List):
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}

