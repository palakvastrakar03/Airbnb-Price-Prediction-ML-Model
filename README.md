# Airbnb Price Prediction – End-to-End Machine Learning System

## Overview
This project presents a **production-ready, end-to-end machine learning pipeline** for predicting Airbnb listing prices using structured data. It demonstrates best practices across **EDA, feature engineering, preprocessing, model training, evaluation, and API deployment readiness**.

The model predicts **`log_price`** to stabilize variance and improve regression performance.

---

## Highlights
- Clean, modular ML pipeline (Scikit-learn Pipelines + ColumnTransformer)
- Strong tabular model (**XGBoost Regressor**) with tuned hyperparameters
- Reproducible evaluation metrics
- Flask API for inference
- GitHub-ready structure with clear separation of concerns

---

## Problem Statement
Airbnb prices vary widely based on:
- Property type and capacity
- Location (city, neighborhood, geo-coordinates)
- Host experience and responsiveness
- Reviews, amenities, and booking policies

Manual pricing is often inaccurate.  
**Objective:** Predict the logarithm of Airbnb listing prices (`log_price`) using supervised ML to enable data-driven pricing.

---

## Dataset
- **Source:** Airbnb Open Dataset  
- **Target Variable:** `log_price`  
- **Type:** Structured tabular data  



---

## Feature Engineering
Engineered features used to improve predictive power:
- **Amenities Count**
- **Host Experience (days since joining)**
- **Days Since Last Review**
- **City-Level Average Price**
- **Location Clusters** (KMeans on latitude & longitude)
- **Host Response Rate** (converted to numeric percentage)
- **Standardized Boolean Features**

---

## Data Preprocessing
- Missing numerical values → median imputation  
- Numerical features → `StandardScaler`  
- Categorical features → `OneHotEncoder(handle_unknown="ignore")`  
- Boolean features → binary encoding  
- Unified preprocessing using **ColumnTransformer**

---

## Model
### Algorithm
**XGBoost Regressor**

### Why XGBoost?
- Captures non-linear relationships
- Handles feature interactions efficiently
- Excellent performance on structured/tabular datasets

### Training Strategy
- Pipeline-based training (preprocessing + model)
- Train/Test split: **80 / 20**
- Tuned hyperparameters for improved generalization

---

## Performance
| Metric | Value |
|------|------|
| RMSE | ~0.37 |
| MAE  | ~0.26 |
| R²   | ~0.73 |

The model explains approximately **73% of the variance** in Airbnb listing prices.

---

## Evaluation Metrics
- **RMSE** – Penalizes large errors
- **MAE** – Interpretable absolute error
- **R² Score** – Explained variance
<img width="740" height="96" alt="Screenshot 2025-12-27 140147" src="https://github.com/user-attachments/assets/1f7dc323-0498-4d19-ba0c-4398642b4e46" />

---


## Project Structure
 ```
Airbnb-Price-Prediction-ML-Model/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebook/
│   └── exploration.ipynb
│
├── models/
│   └── xgb_price_pipeline.pkl
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── model_training.py
    ├── model_evaluation.py
    └── predict.py
 ```

<img width="331" height="597" alt="image" src="https://github.com/user-attachments/assets/f605eda2-8085-4560-8559-344cacc38ef9" />

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/palakvastrakar03/Airbnb-Price-Prediction-ML-Model.gitcd Airbnb-Price-Prediction-ML-Model
2. Install dependencies:
   pip install -r requirements.txt
3. Open and run the notebook:
   jupyter notebook
4. Run the API
   python app.py

   API will be available at: http://127.0.0.1:5000

---

## Technologies Used

Python

Pandas and NumPy

Scikit-learn

XGBoost

Matplotlib and Seaborn

Flask 

Jupyter Notebook

---

##  API Example

POST /predict

Request:
```json
{
  "accommodates": 2,
  "bedrooms": 1,
  "bathrooms": 1,
  "city": "NYC",
  "room_type": "Entire home/apt"
}
```

Response:
```{
  "predicted_log_price": 5.20
}
```

---

## Future Enhancements

- Hyperparameter optimization using Optuna or GridSearchCV
- Model explainability using SHAP values
- Ensemble learning using XGBoost and LightGBM
- Containerization using Docker
- CI/CD pipeline for automated testing and deployment
- Cloud deployment (Render / AWS / Azure / GCP)
- Frontend UI for user-friendly price prediction

---

## Author

Palak Vastrakar

---

## Acknowledgements

Airbnb Open Dataset

Scikit-learn and XGBoost documentation

Open-source Machine Learning community

