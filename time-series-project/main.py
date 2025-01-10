import logging
import os
from src.ets_decomposition import ETSDecomposition
from src.time_series_models import TimeSeriesModels
from src.etl_pipeline import ETLPipeline
from src.model_saving import ModelSaver
from src.hyperparametertune import HyperparameterTuning
from src.logging_config import setup_logging

def main():
    try:
        # Set up logging
        setup_logging(log_file="time_series_pipeline.log")
        logger = logging.getLogger("TimeSeriesPipeline")

        logger.info("Starting the time series pipeline...")

        try:
            # Initialize ModelSaver
            model_saver = ModelSaver(save_dir="time-series-project/models")

            # Ensure the directory exists
            if not os.path.exists("time-series-project/models"):
                os.makedirs("time-series-project/models")
                logger.info("Created models directory.")

            # File path
            logger.info("Starting ETL pipeline...")
            processed_dir = 'time-series-project/data/processed'
            local_raw_file = 'time-series-project/data/raw/final_data.csv'

            # Check if raw file exists
            if not os.path.exists(local_raw_file):
                logger.error(f"Raw data file not found: {local_raw_file}")
                raise FileNotFoundError(f"Raw data file not found: {local_raw_file}")

            etl = ETLPipeline(
                bucket_name='myrawdata7',
                raw_file_name='final_data.csv',
                local_raw_file=local_raw_file,
                processed_dir=processed_dir
            )
            etl.run_pipeline()
            logger.info("ETL pipeline completed successfully.")

            # Perform ETS decomposition
            logger.info("Starting ETS decomposition...")
            processed_file_path = f"{processed_dir}/processed_data.csv"
            ets = ETSDecomposition(file_path=processed_file_path, date_col="Date", value_col="Views")
            ets.load_data()
            components = ets.decompose(period=7)  # Weekly seasonality
            logger.info("ETS decomposition completed successfully.")

            # Use residual or original data for modeling
            time_series_data = components["residual"].dropna()  # Use residuals

            # Hyperparameter Tuning
            logger.info("Starting hyperparameter tuning...")
            tuner = HyperparameterTuning(time_series_data)

            # Tune ARIMA
            arima_params = tuner.tune_arima(p_values=[0, 1, 2], d_values=[0, 1], q_values=[0, 1, 2])
            logger.info(f"Best ARIMA Params: {arima_params}")

            # Tune SARIMA
            sarima_params = tuner.tune_sarima(
                p_values=[0, 1], d_values=[0, 1], q_values=[0, 1],
                P_values=[0, 1], D_values=[0, 1], Q_values=[0, 1], m=7
            )
            logger.info(f"Best SARIMA Params: {sarima_params}")

            # Tune SARIMAX
            exog_data = None  # Replace with actual exogenous data if available
            tuner_with_exog = HyperparameterTuning(time_series_data, exog_train=exog_data)
            sarimax_params = tuner_with_exog.tune_sarimax(
                p_values=[0, 1], d_values=[0, 1], q_values=[0, 1],
                P_values=[0, 1], D_values=[0, 1], Q_values=[0, 1], m=7
            )
            logger.info(f"Best SARIMAX Params: {sarimax_params}")

            # Train ARIMA with best params
            logger.info("Training ARIMA model...")
            models = TimeSeriesModels(time_series_data)
            arima_results = models.train_arima(order=arima_params["best_params"])
            model_saver.save_model(arima_results["model"], "ARIMA_Tuned")
            logger.info("ARIMA model trained and saved successfully.")

            # Train SARIMA with best params
            logger.info("Training SARIMA model...")
            sarima_results = models.train_sarima(order=sarima_params["best_params"][:3],
                                                 seasonal_order=sarima_params["best_params"][3:] + (7,))
            model_saver.save_model(sarima_results["model"], "SARIMA_Tuned")
            logger.info("SARIMA model trained and saved successfully.")

            # Train SARIMAX with best params
            logger.info("Training SARIMAX model...")
            sarimax_results = models.train_sarimax(order=sarimax_params["best_params"][:3],
                                                   seasonal_order=sarimax_params["best_params"][3:] + (7,),
                                                   exog_train=exog_data, exog_test=None)
            model_saver.save_model(sarimax_results["model"], "SARIMAX_Tuned")
            logger.info("SARIMAX model trained and saved successfully.")

        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error initializing the pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()
