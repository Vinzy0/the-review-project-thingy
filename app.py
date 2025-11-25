import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

from sentiment_engine import recommend_products, generate_explanation

app = Flask(__name__)

# Load CNN model
MODEL_PATH = "hair_type_efficientnetb3_finetuned.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ["Straight", "Wavy", "Curly"]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_hair_type(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = [model.predict(img_array, verbose=0) for _ in range(10)]
    avg_pred = np.mean(preds, axis=0)
    hair_type = CLASS_NAMES[np.argmax(avg_pred)]
    confidence = float(np.max(avg_pred) * 100)
    return hair_type, confidence

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = os.path.join("temp_image.jpg")
    file.save(filepath)

    # CNN
    hair_type, confidence = predict_hair_type(filepath)

    # NLP
    recommendations = recommend_products(hair_type)
    explanation = generate_explanation(hair_type, confidence, recommendations)

    return jsonify({
        "hair_type": hair_type,
        "confidence": confidence,
        "recommendations": recommendations,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
