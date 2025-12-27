REQUIRED_FIELDS = [
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "city",
    "cleaning_fee",
    "instant_bookable"
]


from flask import Flask, request, jsonify
from src.predict import predict_price

app = Flask(__name__)

@app.route("/")
def home():
    return {"status": "Airbnb Price Prediction API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate input
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    prediction = predict_price(data)
    return jsonify({
        "predicted_log_price": prediction
    })


# 👇 ADD THIS LINE
print(app.url_map)

if __name__ == "__main__":
    app.run(debug=True)
