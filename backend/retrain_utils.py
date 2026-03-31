import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from backend.model_utils import train_model

def retrain_with_selected_features(df, target_col, model_choice, split_type, selected_features):
    """
    Retrain a model using only selected features.
    Returns: model, data dict, scaler, rmse
    """
    reduced_df = df[selected_features + [target_col]]
    model, data, metrics, scaler = train_model(
        reduced_df, target_col, model_choice, split_type=split_type
    )

    # Compute RMSE for error margin
    y_true = data["y_test"]
    y_hat = model.predict(data["X_test_scaled"])
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))

    return model, data, scaler, rmse


def predict_with_inputs(model, scaler, selected_features, input_values, rmse):
    """
    Predict target value given manual inputs.
    Applies scaler if available, returns prediction ± rmse.
    """
    X_manual = pd.DataFrame([input_values])[selected_features]

    if scaler is not None:
        X_manual_scaled = scaler.transform(X_manual)
    else:
        X_manual_scaled = X_manual.values

    y_pred = float(model.predict(X_manual_scaled)[0])
    return y_pred, rmse