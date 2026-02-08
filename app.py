from fastapi import FastAPI
import numpy as np
import tensorflow as tf

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
