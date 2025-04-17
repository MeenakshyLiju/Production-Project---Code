import streamlit as st
import pandas as pd
import boto3
import json
from io import StringIO, BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# AWS Setup
bucket_name = "sagemaker-eu-west-2-047719636927"
region = "eu-west-2"
s3 = boto3.client("s3", region_name=region)
runtime = boto3.client("sagemaker-runtime", region_name=region)

# Define endpoint names
xgb_demand_endpoint = "xgboost-demand-prod"
xgb_price_endpoint  = "xgboost-price-prod"

# Load diffusion predictions from S3
@st.cache_data
def load_diffusion_predictions():
    files = {
        "diffusion_demand": "diffusion-models/diffusion_demand_predictions.csv",
        "diffusion_price": "diffusion-models/diffusion_price_predictions.csv"
    }
    predictions = {}
    for name, path in files.items():
        obj = s3.get_object(Bucket=bucket_name, Key=path)
        df = pd.read_csv(obj["Body"])
        if "Date" in df.columns:
            df = df.drop(columns=["Date"])
        predictions[name] = df
    return predictions

# Function to load scaler files from S3
def load_scaler_from_s3(scaler_key):
    obj = s3.get_object(Bucket=bucket_name, Key=scaler_key)
    scaler = joblib.load(BytesIO(obj["Body"].read()))
    return scaler

# Load the X scalers for demand and price
scaler_X_demand = load_scaler_from_s3("xgboost-models/scaler_X_demand.joblib")
scaler_X_price  = load_scaler_from_s3("xgboost-models/scaler_X_price.joblib")

# Also load the corresponding Y scalers
scaler_y_demand = load_scaler_from_s3("xgboost-models/scaler_y_demand.joblib")
scaler_y_price  = load_scaler_from_s3("xgboost-models/scaler_y_price.joblib")

# Function to call a SageMaker endpoint given a DataFrame input
def get_endpoint_predictions(endpoint_name, df):
    # Convert the DataFrame to CSV string
    payload = df.to_csv(header=True, index=False)
    response = runtime.invoke_endpoint(
         EndpointName=endpoint_name,
         ContentType="text/csv",
         Body=payload
    )
    result = response['Body'].read().decode('utf-8')
    pred_dict = json.loads(result)
    return pred_dict

# Streamlit UI Setup
st.set_page_config(page_title="E-Commerce Price and Demand Forecasting Dashboard", layout="wide")
st.title("🛍️ E-Commerce Price & Demand Forecasting Dashboard")
st.markdown("📂 **Upload your CSV file to begin**")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Read the raw CSV
    raw_df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
    st.subheader("🔍 Uploaded Data Sample Preview")
    st.dataframe(raw_df.head())
    # Drop any existing "Date" column 
    if "Date" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Date"])
    
    # Check required date feature columns
    required_date_cols = ["Year", "Month", "Day", "Weekday"]
    missing = [col for col in required_date_cols if col not in raw_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}. Please ensure your file includes {', '.join(required_date_cols)}.")
    else:
        # Compute the Date column using Year, Month, Day
        raw_df["Date"] = pd.to_datetime(raw_df[["Year", "Month", "Day"]])
        
        # Define base features used in training
        base_features = [
            "Year", "Month", "Day", "Weekday", "Product_ID",
            "Marketing_Campaign", "Seasonal_Trend", "Stock_Availability",
            "Base_Sales", "Marketing_Effect", "Seasonal_Effect",
            "Discount", "Competitor_Price", "Public_Holiday"
        ]

        # XGBoost Demand Model Processing
        demand_feature_cols = base_features + ['Price']  # columns used for training / scaling
        xgb_demand_input = raw_df[demand_feature_cols + ['Demand']].copy()
        
        # Scale the predictor columns using scaler_X_demand
        xgb_demand_input[demand_feature_cols] = scaler_X_demand.transform(xgb_demand_input[demand_feature_cols])
        # Also scale "Demand" using scaler_y_demand
        xgb_demand_input['Demand'] = scaler_y_demand.transform(xgb_demand_input[['Demand']])

        # XGBoost Price Model Processing
        price_feature_cols = base_features + ['Demand']
        xgb_price_input = raw_df[price_feature_cols + ['Price']].copy()
        # Scale the predictor columns using scaler_X_price
        xgb_price_input[price_feature_cols] = scaler_X_price.transform(xgb_price_input[price_feature_cols])
        # Also scale "Price" using scaler_y_price
        xgb_price_input['Price'] = scaler_y_price.transform(xgb_price_input[['Price']])
        
        missing_demand = [col for col in demand_feature_cols if col not in raw_df.columns]
        missing_price  = [col for col in price_feature_cols if col not in raw_df.columns]
        if missing_demand:
            st.error(f"Missing columns for XGBoost Demand Prediction: {', '.join(missing_demand)}")
        elif missing_price:
            st.error(f"Missing columns for XGBoost Price Prediction: {', '.join(missing_price)}")
        else:
            # Get XGBoost predictions from endpoints using the scaled input data.
            with st.spinner("Generating XGBoost predictions."):
                xgb_demand_response = get_endpoint_predictions(xgb_demand_endpoint, xgb_demand_input)
                xgb_price_response  = get_endpoint_predictions(xgb_price_endpoint, xgb_price_input)
            
            # Get predictions from the endpoint responses.
            xgb_demand_preds = xgb_demand_response.get("predictions", [])
            xgb_price_preds  = xgb_price_response.get("predictions", [])
            
            # Load diffusion predictions from S3
            diffusion_preds = load_diffusion_predictions()
            for name, df in diffusion_preds.items():
                if set(["Year", "Month", "Day"]).issubset(df.columns):
                    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
                elif "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
            # Results
            results_df = pd.DataFrame({
                "Date": raw_df["Date"],
                "Demand": raw_df["Demand"],
                "Price": raw_df["Price"],
                "XGBoost_Predicted_Demand": pd.Series(xgb_demand_preds).astype(float),
                "XGBoost_Predicted_Price": pd.Series(xgb_price_preds).astype(float),
                "Diffusion_Predicted_Demand": diffusion_preds["diffusion_demand"]["Diffusion_Predicted_Demand"].astype(float),
                "Diffusion_Predicted_Price": diffusion_preds["diffusion_price"]["Diffusion_Predicted_Price"].astype(float)
            })
            results_df["Month"] = results_df["Date"].dt.to_period("M").astype(str)
            
            st.subheader("📈 Prediction Results")
            st.dataframe(results_df.head(10).style.format({col: "{:.2f}" for col in results_df.select_dtypes(include=[np.number]).columns}))
            
            st.subheader("📊 Model Evaluation Metrics")

            model_metrics = {
                "XGBoost Demand":  {"MAE": 0.0191, "MSE": 0.0009, "RMSE": 0.0302, "R2": 0.9952},
                "XGBoost Price": xgb_price_response.get("evaluation_metrics", {}),
                "Diffusion Demand": {"MAE": 0.1079, "MSE": 0.0135, "RMSE": 0.1164, "R2": 0.9856},
                "Diffusion Price": {"MAE": 1.9258, "MSE": 6.5887, "RMSE": 2.5669, "R2": 0.9837}
            }
            st.markdown("""
            ### 🧠 Understanding the Metrics
            - **MAE (Mean Absolute Error)**: Average prediction error. Lower = better.
            - **MSE (Mean Squared Error)**: Measures how close predictions are to reality. Lower = better.
            - **RMSE (Root Mean Squared Error)**: Square root of MSE. Lower = better.
            - **R² (R-squared)**: Proportion of variance explained. Closer to 1 = better.
            """)
            st.dataframe(pd.DataFrame(model_metrics).T.style.format("{:.4f}"))
            
            st.subheader("📅 Monthly Demand Forecast (Line Chart)")
            fig1, ax1 = plt.subplots(figsize=(6,4))
            monthly_demand = results_df.groupby("Month")[["Demand", "XGBoost_Predicted_Demand", "Diffusion_Predicted_Demand"]].mean()
            sns.lineplot(data=monthly_demand, ax=ax1)
            ax1.set_ylabel("Average Monthly Demand")
            ax1.set_xlabel("Month")
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            col_left, col_mid, col_right = st.columns([2,5,2])
            with col_mid:
                st.pyplot(fig1, use_container_width=False)

            
            st.subheader("💰 Monthly Price Forecast (Line Chart)")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            monthly_price = results_df.groupby("Month")[["Price", "XGBoost_Predicted_Price", "Diffusion_Predicted_Price"]].mean()
            sns.lineplot(data=monthly_price, ax=ax2)
            ax2.set_ylabel("Average Monthly Price")
            ax2.set_xlabel("Month")
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            col_left, col_mid, col_right = st.columns([2,5,2])
            with col_mid:
                st.pyplot(fig2, use_container_width=False)
            
            st.subheader("🎯 Actual vs Predicted: Demand")
            fig3, ax3 = plt.subplots(figsize=(5,5))
            sns.scatterplot(x="Demand", y="XGBoost_Predicted_Demand", data=results_df, label="XGBoost", alpha=0.5, ax=ax3)
            sns.scatterplot(x="Demand", y="Diffusion_Predicted_Demand", data=results_df, label="Diffusion", alpha=0.5, ax=ax3)
            ax3.plot([0, results_df["Demand"].max()], [0, results_df["Demand"].max()], linestyle="--", color="red")
            ax3.set_xlabel("Actual Demand")
            ax3.set_ylabel("Predicted Demand")
            ax3.legend()
            col_left, col_mid, col_right = st.columns([2,5,2])
            with col_mid:
                st.pyplot(fig3, use_container_width=False)

            
            st.subheader("📦 Price Distribution by Model")
            price_melted = results_df[["Price", "XGBoost_Predicted_Price", "Diffusion_Predicted_Price"]].melt(var_name="Type", value_name="Price")

            custom_palette = {
                "Price": "#1f77b4",
                "XGBoost_Predicted_Price": "#ff7f0e",
                "Diffusion_Predicted_Price": "#2ca02c" 
            }
            
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sns.boxplot(
                data=price_melted, 
                x="Type", 
                y="Price", 
                ax=ax4,
                palette=custom_palette
            )
            ax4.tick_params(axis='x', labelrotation=30)  
            ax4.set_ylabel("Price (%)")
            ax4.set_xlabel("")
            col_left, col_mid, col_right = st.columns([2,5,2])
            with col_mid:
                st.pyplot(fig4, use_container_width=False)

else:
    st.warning("Please upload a CSV file")
