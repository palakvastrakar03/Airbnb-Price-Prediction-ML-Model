import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model using RMSE, MAE, and R2 score
    """
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
