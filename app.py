import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from typing import List
import gdown

app = FastAPI()

MODEL_PATH = "model.h5"

# 🔹 Your Google Drive FILE ID (not full link)
FILE_ID = "1e8aOjV6A9B1ZMakxfcdfSTrq9mj4gglh"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )
        print("Model downloaded")

download_model()

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model download failed")

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "ML server is running"}

@app.post("/predict")
def predict(data: List):
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}

