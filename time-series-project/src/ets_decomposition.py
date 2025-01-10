import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


class ETSDecomposition:
    def __init__(self, file_path, date_col, value_col):
        """
        Initialize the ETSDecomposition class.
        
        Args:
            file_path (str): Path to the CSV file.
            date_col (str): Column name for the date.
            value_col (str): Column name for the value.
        """
        self.file_path = file_path
        self.date_col = date_col
        self.value_col = value_col
        self.data = None

    def load_data(self):
        """Loads the data from the CSV file."""
        self.data = pd.read_csv(self.file_path)
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        self.data.set_index(self.date_col, inplace=True)
        print("Data loaded successfully.")

    def decompose(self, period, model='additive'):
        """
        Perform ETS decomposition on the time series data.
        
        Args:
            period (int): The seasonal period (e.g., 7 for weekly data).
            model (str): Type of decomposition ('additive' or 'multiplicative').
        
        Returns:
            dict: A dictionary containing trend, seasonality, and residual components.
        """
        decomposition = seasonal_decompose(self.data[self.value_col], model=model, period=period)
        decomposition.plot()
        plt.show()
        plt.savefig(fname='time-series-project/picture')
        
        print("ETS Decomposition completed.")
        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid
        }
