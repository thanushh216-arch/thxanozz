import os
import gdown
from tensorflow.keras.models import load_model
from fastapi import FastAPI


MODEL_PATH = "model.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1zs-qoEU2l9hcgo8udYZ6CvdfJV4m3iL0"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

print("✅ Loading model...")
model = load_model(MODEL_PATH)
print("🚀 Model loaded successfully")

def predict(data: List):
    array = np.array(data)
    prediction = model.predict(array)
    return {"prediction": prediction.tolist()}
