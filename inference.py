import os
import joblib
import numpy as np
import pandas as pd
import json
import xgboost as xgb
from typing import Dict, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def model_fn(model_dir: str) -> Tuple[xgb.XGBModel, list, str, object]:
    base_features = [
        "Year", "Month", "Day", "Weekday", "Product_ID",
        "Marketing_Campaign", "Seasonal_Trend", "Stock_Availability",
        "Base_Sales", "Marketing_Effect", "Seasonal_Effect",
        "Discount", "Competitor_Price", "Public_Holiday"
    ]
    
    try:
        if 'xgb_demand.joblib' in os.listdir(model_dir):
            model = joblib.load(os.path.join(model_dir, 'xgb_demand.joblib'))
            expected_features = base_features + ['Price']
            model_type = 'demand'
            # Load the target scaler used in training for the Demand values
            scaler_y_path = os.path.join(model_dir, 'scaler_y_demand.joblib')
            scaler_y = joblib.load(scaler_y_path)
        elif 'xgb_price.joblib' in os.listdir(model_dir):
            model = joblib.load(os.path.join(model_dir, 'xgb_price.joblib'))
            expected_features = base_features + ['Demand']
            model_type = 'price'
            # Load the target scaler used in training for the Price values.
            scaler_y_path = os.path.join(model_dir, 'scaler_y_price.joblib')
            scaler_y = joblib.load(scaler_y_path)
        else:
            raise ValueError("No valid model file found in the model directory.")
        
        if hasattr(model, 'predictor'):
            model.set_params(predictor='cpu')
        if hasattr(model, 'gpu_id'):
            model.set_params(gpu_id=-1)
            
        return model, expected_features, model_type, scaler_y
        
    except Exception as e:
        raise ValueError(f"Model loading failed: {str(e)}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def predict_fn(input_data: Union[dict, pd.DataFrame], 
               model_data: Tuple[xgb.XGBModel, list, str, object]) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
    model, expected_features, model_type, scaler_y = model_data
    df = pd.DataFrame(input_data)
    ground_truth = None
    if model_type == 'demand' and 'Demand' in df.columns:
        ground_truth = df['Demand'].values
    elif model_type == 'price' and 'Price' in df.columns:
        ground_truth = df['Price'].values
    
    # Validate the presence of all expected features.
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the input data: {missing_cols}")

    X = df[expected_features].astype(np.float32)
    
    # Prediction 
    try:
        preds = model.predict(X.values)
    except Exception as e:
        try:
            dmatrix = xgb.DMatrix(X.values, feature_names=expected_features)
            preds = model.predict(dmatrix)
        except Exception as inner_e:
            from xgboost import Booster
            if isinstance(model, Booster):
                dmatrix = xgb.DMatrix(X.values)
                preds = model.predict(dmatrix)
            else:
                safe_model = xgb.XGBRegressor()
                safe_model._Booster = model.get_booster()
                preds = safe_model.predict(X.values)
    
    # Inverse-transform the predictions to the original scale.
    try:
        preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    except Exception as e:
        raise ValueError(f"Inverse transform of predictions failed: {str(e)}")
    
    metrics = {}
    if ground_truth is not None:

        try:
            ground_truth_orig = scaler_y.inverse_transform(np.array(ground_truth).reshape(-1, 1)).flatten()
        except Exception as e:

            ground_truth_orig = ground_truth
            
        metrics = compute_metrics(ground_truth_orig, preds)
        
    return {"predictions": preds, "metrics": metrics}


def input_fn(request_body: str, content_type: str = 'application/json') -> Union[dict, pd.DataFrame]:

    if content_type == 'application/json':
        data = json.loads(request_body)
        if not isinstance(data, (dict, list)):
            raise ValueError("JSON input must be a dictionary or a list of records.")
        return data
    elif content_type == 'text/csv':
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(request_body))
            if df.empty:
                raise ValueError("CSV data is empty.")
            return df.to_dict('records')
        except Exception as e:
            raise ValueError(f"CSV parsing failed: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}. Use 'application/json' or 'text/csv'.")


def output_fn(prediction: Dict[str, Union[np.ndarray, Dict[str, float]]], 
              accept: str = 'application/json') -> Tuple[str, str]:

    try:
        output = {
            "predictions": prediction["predictions"].tolist() if isinstance(prediction["predictions"], np.ndarray)
                           else prediction["predictions"],
            "evaluation_metrics": prediction["metrics"]
        }
        if accept == 'application/json':
            return json.dumps(output), accept
        elif accept == 'text/csv':
            return pd.DataFrame({"predictions": prediction["predictions"]}).to_csv(index=False), accept
        else:
            raise ValueError(f"Unsupported accept type: {accept}")
    except Exception as e:
        raise ValueError(f"Output formatting failed: {str(e)}")
