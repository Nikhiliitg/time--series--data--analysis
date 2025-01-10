import streamlit as st
from src.ets_decomposition import ETSDecomposition
from src.time_series_models import TimeSeriesModels
from src.etl_pipeline import ETLPipeline
from src.model_saving import ModelSaver
from src.hyperparametertune import HyperparameterTuning
import logging

# Title of the app
st.title("Time Series Analysis and Forecasting")

# Sidebar inputs
st.sidebar.header("Configuration")
bucket_name = st.sidebar.text_input("S3 Bucket Name", "myrawdata7")
raw_file_name = st.sidebar.text_input("Raw File Name", "final_data.csv")
processed_dir = "time-series-project/data/processed"
local_raw_file = "time-series-project/data/raw/final_data.csv"
save_dir = "time-series-project/models"

# Run ETL Pipeline
if st.sidebar.button("Run ETL Pipeline"):
    st.write("Running ETL Pipeline...")
    try:
        etl = ETLPipeline(bucket_name, raw_file_name, local_raw_file, processed_dir)
        etl.run_pipeline()
        st.success("ETL Pipeline completed successfully!")
    except Exception as e:
        st.error(f"ETL Pipeline failed: {e}")

# Perform ETS Decomposition
if st.sidebar.button("Perform ETS Decomposition"):
    st.write("Performing ETS Decomposition...")
    try:
        processed_file_path = f"{processed_dir}/processed_data.csv"
        ets = ETSDecomposition(file_path=processed_file_path, date_col="Date", value_col="Views")
        ets.load_data()
        components = ets.decompose(period=7)  # Weekly seasonality
        st.success("ETS Decomposition completed successfully!")
        st.write("Decomposition Components:")
        st.dataframe(components)
    except Exception as e:
        st.error(f"ETS Decomposition failed: {e}")

# Train Models
if st.sidebar.button("Train Models"):
    st.write("Training Models...")
    try:
        processed_file_path = f"{processed_dir}/processed_data.csv"
        ets = ETSDecomposition(file_path=processed_file_path, date_col="Date", value_col="Views")
        ets.load_data()
        components = ets.decompose(period=7)
        time_series_data = components["residual"].dropna()  # Use residuals

        models = TimeSeriesModels(time_series_data)

        # Train ARIMA
        arima_results = models.train_arima(order=(1, 1, 1))  # Initial ARIMA params
        model_saver = ModelSaver(save_dir)
        model_saver.save_model(arima_results["model"], "ARIMA_Initial")
        st.write("ARIMA model trained and saved.")

        # Train SARIMA
        sarima_results = models.train_sarima(order=(1, 1, 1), seasonal_order=(0, 1, 1, 7))  # Initial SARIMA params
        model_saver.save_model(sarima_results["model"], "SARIMA_Initial")
        st.write("SARIMA model trained and saved.")

        # Train SARIMAX
        exog_data = None  # Replace with actual exogenous data
        sarimax_results = models.train_sarimax(
            order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), exog_train=exog_data, exog_test=None
        )
        model_saver.save_model(sarimax_results["model"], "SARIMAX_Initial")
        st.write("SARIMAX model trained and saved.")
        st.success("All models trained successfully!")
    except Exception as e:
        st.error(f"Model training failed: {e}")

# Hyperparameter Tuning
if st.sidebar.button("Hyperparameter Tuning"):
    st.write("Tuning Hyperparameters...")
    try:
        processed_file_path = f"{processed_dir}/processed_data.csv"
        ets = ETSDecomposition(file_path=processed_file_path, date_col="Date", value_col="Views")
        ets.load_data()
        components = ets.decompose(period=7)
        time_series_data = components["residual"].dropna()  # Use residuals

        tuner = HyperparameterTuning(time_series_data)

        # Tune ARIMA
        st.write("Tuning ARIMA parameters...")
        arima_params = tuner.tune_arima(p_values=[0, 1, 2], d_values=[0, 1], q_values=[0, 1, 2])
        st.write(f"Best ARIMA Parameters: {arima_params}")

        # Tune SARIMA
        st.write("Tuning SARIMA parameters...")
        sarima_params = tuner.tune_sarima(
            p_values=[0, 1], d_values=[0, 1], q_values=[0, 1],
            P_values=[0, 1], D_values=[0, 1], Q_values=[0, 1], m=7
        )
        st.write(f"Best SARIMA Parameters: {sarima_params}")

        # Tune SARIMAX
        st.write("Tuning SARIMAX parameters...")
        exog_data = None  # Replace with actual exogenous data
        tuner_with_exog = HyperparameterTuning(time_series_data, exog_train=exog_data)
        sarimax_params = tuner_with_exog.tune_sarimax(
            p_values=[0, 1], d_values=[0, 1], q_values=[0, 1],
            P_values=[0, 1], D_values=[0, 1], Q_values=[0, 1], m=7
        )
        st.write(f"Best SARIMAX Parameters: {sarimax_params}")

        st.success("Hyperparameter tuning completed successfully!")
    except Exception as e:
        st.error(f"Hyperparameter tuning failed: {e}")
