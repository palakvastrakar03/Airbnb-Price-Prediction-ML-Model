REQUIRED_COLUMNS = [
    'property_type', 'room_type', 'accommodates', 'bathrooms',
    'bedrooms', 'beds', 'city', 'cleaning_fee', 'instant_bookable',
    'latitude', 'longitude', 'host_response_rate',
    'host_identity_verified', 'host_has_profile_pic',
    'review_scores_rating', 'number_of_reviews',
    'amenities_count', 'bed_type', 'cancellation_policy',
    'days_since_last_review', 'host_experience_days',
    'city_avg_price', 'location_cluster'
]



import joblib
import pandas as pd
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_price_pipeline.pkl")

# Load model ONCE
model = joblib.load(MODEL_PATH)

def predict_price(input_data: dict):
    """
    Predict log price for a single input.
    Missing features are auto-filled with defaults.
    """

    # Default values for engineered features
    defaults = {
        'latitude': 0.0,
        'longitude': 0.0,
        'host_response_rate': 0.5,
        'host_identity_verified': 0,
        'host_has_profile_pic': 1,
        'review_scores_rating': 90,
        'number_of_reviews': 0,
        'amenities_count': 5,
        'bed_type': 'Real Bed',
        'cancellation_policy': 'flexible',
        'days_since_last_review': 365,
        'host_experience_days': 365,
        'city_avg_price': 4.5,
        'location_cluster': 0
    }

    # Merge user input with defaults
    final_input = {**defaults, **input_data}

    # Ensure all required columns exist
    df = pd.DataFrame([final_input])

    prediction = model.predict(df)
    return float(prediction[0])
