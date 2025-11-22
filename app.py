import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from flask import Flask, request

app = Flask(__name__)

MODEL_PATH = "model/hair_type_efficientnetb3_finetuned.h5"
IMG_SIZE = (300, 300)
CLASS_NAMES = ["Straight", "Wavy", "Curly"]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_hair_type(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = [model.predict(img_array, verbose=0) for _ in range(10)]
    avg_pred = np.mean(preds, axis=0)

    predicted_class = CLASS_NAMES[np.argmax(avg_pred)]
    confidence = float(np.max(avg_pred) * 100)
    return predicted_class, confidence

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files['image']
    filepath = "temp_image.jpg"
    file.save(filepath)

    hair_type, confidence = predict_hair_type(filepath)

    recommendations = recommend_products(hair_type)

    final_output = generate_explanation(hair_type, confidence, recommendations)

    return {
        "hair_type": hair_type,
        "confidence": confidence,
        "recommendations": recommendations,
        "explanation": final_output
    }
