import os
from flask import Flask, request, jsonify
# Import your logic from the other file
from sentiment_engine import recommend_products, generate_explanation

app = Flask(__name__)

# This is the API Endpoint
# URL: http://<your-ip>:5000/recommend
# Method: POST (Because we are sending data to it)
@app.route("/recommend", methods=["POST"])
def recommend():
    # 1. RECEIVE THE ORDER
    # The app sends JSON data like: {"hair_type": "Curly", "confidence": 95.5}
    data = request.get_json()
    
    # Check if they actually sent the data
    if not data or 'hair_type' not in data:
        return jsonify({"error": "Please provide a hair_type"}), 400

    hair_type = data['hair_type']
    confidence = data.get('confidence', 0.0) # Default to 0 if not provided

    # 2. TELL THE KITCHEN (Run your Python Logic)
    # This calls the functions you already wrote in sentiment_engine.py
    recommendations = recommend_products(hair_type)
    explanation = generate_explanation(hair_type, confidence, recommendations)

    # 3. SERVE THE DISH (Return JSON)
    # We pack everything into a nice dictionary and convert it to JSON
    response = {
        "status": "success",
        "hair_type_analyzed": hair_type,
        "explanation": explanation,
        "recommendations": recommendations
    }
    
    return jsonify(response)

if __name__ == "__main__":
    # host='0.0.0.0' means "listen to anyone on this wifi network"
    app.run(host='0.0.0.0', port=5000, debug=True)