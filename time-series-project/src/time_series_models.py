import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


class TimeSeriesModels:
    def __init__(self, data, test_size=0.2):
        """
        Initialize the TimeSeriesModels class.
        
        Args:
            data (pd.Series): Time series data.
            test_size (float): Proportion of data for testing (default 0.2).
        """
        self.data = data
        self.train = None
        self.test = None
        self.test_size = test_size
        self.split_data()

    def split_data(self):
        """Split the data into train and test sets."""
        n_test = int(len(self.data) * self.test_size)
        self.train = self.data[:-n_test]
        self.test = self.data[-n_test:]
        print("Train-test split completed.")

    def train_arima(self, order):
        """
        Train an ARIMA model.
        
        Args:
            order (tuple): ARIMA order (p, d, q).
        
        Returns:
            dict: Predictions and the fitted model.
        """
        print("Training ARIMA model...")
        model = ARIMA(self.train, order=order)
        fitted_model = model.fit()
        predictions = fitted_model.forecast(steps=len(self.test))
        return {"predictions": predictions, "model": fitted_model}

    def train_sarima(self, order, seasonal_order):
        """
        Train a SARIMA model.
        
        Args:
            order (tuple): ARIMA order (p, d, q).
            seasonal_order (tuple): Seasonal order (P, D, Q, m).
        
        Returns:
            dict: Predictions and the fitted model.
        """
        print("Training SARIMA model...")
        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        predictions = fitted_model.forecast(steps=len(self.test))
        return {"predictions": predictions, "model": fitted_model}

    def train_sarimax(self, order, seasonal_order, exog_train, exog_test):
        """
        Train a SARIMAX model.
        
        Args:
            order (tuple): ARIMA order (p, d, q).
            seasonal_order (tuple): Seasonal order (P, D, Q, m).
            exog_train (pd.DataFrame): Exogenous variables for training.
            exog_test (pd.DataFrame): Exogenous variables for testing.
        
        Returns:
            dict: Predictions and the fitted model.
        """
        print("Training SARIMAX model...")
        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order, exog=exog_train)
        fitted_model = model.fit(disp=False)
        predictions = fitted_model.forecast(steps=len(self.test), exog=exog_test)
        return {"predictions": predictions, "model": fitted_model}

    def evaluate(self, predictions, model_name):
        """
        Evaluate the model using Mean Absolute Error (MAE).
        
        Args:
            predictions (pd.Series): Predictions from the model.
            model_name (str): Name of the model.
        """
        mae = mean_absolute_error(self.test, predictions)
        print(f"{model_name} Mean Absolute Error (MAE): {mae}")
        return mae
