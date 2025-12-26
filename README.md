# Airbnb Price Prediction – Machine Learning Model

## Project Overview
This project implements an end-to-end Machine Learning pipeline to predict Airbnb listing prices based on property characteristics, host details, location attributes, and review metrics. The goal is to estimate optimal pricing using data-driven techniques.

The project includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training and Evaluation
- Performance Analysis

---

## Problem Statement
Airbnb listing prices vary significantly depending on several factors such as:
- Property type and capacity
- Geographic location
- Host experience and responsiveness
- Reviews and amenities

Manual pricing often results in suboptimal estimates. This project aims to predict the logarithm of the listing price (`log_price`) using supervised machine learning models.

---

## Dataset
- Source: Airbnb Open Dataset
- Target Variable: `log_price`
- Data Type: Structured tabular data

The dataset is loaded using a public URL or local path and is not stored directly in the repository to avoid large file size issues.

---

## Feature Engineering
The following engineered features were used:
- Amenities Count: Number of amenities offered by the property
- Host Experience Days: Number of days since the host joined Airbnb
- Days Since Last Review
- City Average Price: Mean listing price per city
- Location Clusters: KMeans clustering using latitude and longitude
- Host Response Rate: Converted to numeric percentage
- Boolean Feature Standardization

---

## Data Preprocessing
- Missing numerical values filled using median values
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Boolean features converted to binary format
- Pipeline-based preprocessing using `ColumnTransformer`

---

## Machine Learning Model
- Algorithm: XGBoost Regressor
- Reason for Selection: Handles non-linear relationships and feature interactions efficiently
- Training Strategy:
  - Pipeline-based training
  - Train-test split (80:20)
  - Tuned hyperparameters for improved performance

---

## Model Performance

| Metric | Value |
|------|------|
| RMSE | ~0.37 |
| MAE  | ~0.26 |
| R² Score | ~0.73 |

The model explains approximately 73 percent of the variance in Airbnb listing prices.

---

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

These metrics provide a balanced evaluation of model accuracy and robustness.

---

## Project Structure

Airbnb-Price-Prediction-ML-Model/

├── Airbnb Price Prediction.ipynb
├── README.md
├── requirements.txt
|── .gitignore



---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Airbnb-Price-Prediction-ML-Model.git
2. Install dependencies:
   pip install -r requirements.txt
3. Open and run the notebook:
   jupyter notebook

## Technologies Used

Python

Pandas and NumPy

Scikit-learn

XGBoost

Matplotlib and Seaborn

Jupyter Notebook

## Future Enhancements

Hyperparameter optimization using GridSearch or Optuna

Model deployment using Flask or FastAPI

Ensemble modeling using XGBoost and LightGBM

Feature importance visualization

Model explainability using SHAP

## Author

Palak Vastrakar

## Acknowledgements

Airbnb Open Dataset

Scikit-learn and XGBoost documentation

Open-source Machine Learning community

