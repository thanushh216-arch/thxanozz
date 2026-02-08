import os
import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
import gdown

app = FastAPI()

MODEL_PATH = "model.pkl"
FILE_ID = "PASTE_YOUR_FILE_ID_HERE"

def download_model():
    print("Current directory:", os.getcwd())
    print("Files before download:", os.listdir("."))

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

    print("Files after download:", os.listdir("."))

download_model()

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model.pkl NOT FOUND after download")

print("✅ model.pkl found, loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "ML server is running"}

@app.post("/predict")
def predict(data: List):
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}
