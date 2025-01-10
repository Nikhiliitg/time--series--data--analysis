import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np


class HyperparameterTuning:
    def __init__(self, time_series_data, exog_train=None):
        """
        Initialize the HyperparameterTuning class.

        Args:
            time_series_data (pd.Series): The time series data to train on.
            exog_train (pd.DataFrame): Exogenous variables for SARIMAX, default is None.
        """
        self.data = time_series_data
        self.exog_train = exog_train

    def tune_arima(self, p_values, d_values, q_values):
        """
        Perform hyperparameter tuning for ARIMA.

        Args:
            p_values (list): List of values for the AR parameter.
            d_values (list): List of values for the differencing parameter.
            q_values (list): List of values for the MA parameter.

        Returns:
            dict: Best parameters and model performance.
        """
        best_score, best_cfg = float("inf"), None
        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(self.data, order=(p, d, q)).fit()
                predictions = model.predict(start=len(self.data), end=len(self.data) + 10)
                mse = mean_squared_error(self.data[-len(predictions):], predictions)
                if mse < best_score:
                    best_score, best_cfg = mse, (p, d, q)
            except Exception as e:
                continue
        return {"best_score": best_score, "best_params": best_cfg}

    def tune_sarima(self, p_values, d_values, q_values, P_values, D_values, Q_values, m):
        """
        Perform hyperparameter tuning for SARIMA.

        Args:
            p_values, d_values, q_values (list): ARIMA parameters for SARIMA.
            P_values, D_values, Q_values (list): Seasonal ARIMA parameters.
            m (int): Seasonal period.

        Returns:
            dict: Best parameters and model performance.
        """
        best_score, best_cfg = float("inf"), None
        for p, d, q, P, D, Q in itertools.product(
            p_values, d_values, q_values, P_values, D_values, Q_values
        ):
            try:
                model = SARIMAX(
                    self.data, order=(p, d, q), seasonal_order=(P, D, Q, m)
                ).fit()
                predictions = model.predict(start=len(self.data), end=len(self.data) + 10)
                mse = mean_squared_error(self.data[-len(predictions):], predictions)
                if mse < best_score:
                    best_score, best_cfg = mse, (p, d, q, P, D, Q)
            except Exception as e:
                continue
        return {"best_score": best_score, "best_params": best_cfg}

    def tune_sarimax(self, p_values, d_values, q_values, P_values, D_values, Q_values, m):
        """
        Perform hyperparameter tuning for SARIMAX.

        Args:
            p_values, d_values, q_values (list): ARIMA parameters for SARIMAX.
            P_values, D_values, Q_values (list): Seasonal ARIMA parameters.
            m (int): Seasonal period.

        Returns:
            dict: Best parameters and model performance.
        """
        best_score, best_cfg = float("inf"), None
        for p, d, q, P, D, Q in itertools.product(
            p_values, d_values, q_values, P_values, D_values, Q_values
        ):
            try:
                model = SARIMAX(
                    self.data, exog=self.exog_train, order=(p, d, q), seasonal_order=(P, D, Q, m)
                ).fit()
                predictions = model.predict(start=len(self.data), end=len(self.data) + 10)
                mse = mean_squared_error(self.data[-len(predictions):], predictions)
                if mse < best_score:
                    best_score, best_cfg = mse, (p, d, q, P, D, Q)
            except Exception as e:
                continue
        return {"best_score": best_score, "best_params": best_cfg}
