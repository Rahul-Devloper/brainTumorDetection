from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import cv2
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

# Model + threshold (can be overridden with env vars in Docker/cloud)
MODEL_PATH = os.getenv(
    "MODEL_PATH", "../notebooks/api/model/brain_mri_model.h5")
THRESHOLD = float(os.getenv("THRESHOLD", "0.05"))

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Set MODEL_PATH env var to the correct path."
    )

model = tf.keras.models.load_model(MODEL_PATH, compile=False)



def preprocess(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Unable to decode image")
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,28,28,1)
    return img


@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": 'Missing form field "file"'}), 400
    x = preprocess(request.files["file"].read())
    proba = float(model.predict(x, verbose=0).reshape(-1)[0])
    label_id = 1 if proba >= THRESHOLD else 0
    label = "tumor" if label_id == 1 else "no_tumor"
    return jsonify({
        "probability_tumor": proba,
        "threshold": THRESHOLD,
        "label_id": label_id,
        "label_name": label
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
