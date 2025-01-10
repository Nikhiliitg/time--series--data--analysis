import os
import pickle

class ModelSaver:
    def __init__(self, save_dir="time-series-project/models"):
        """
        Initialize the ModelSaver class.
        
        Args:
            save_dir (str): Directory where models will be saved.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_model(self, model, model_name):
        """
        Save the model to a file.
        
        Args:
            model: The trained model object to save.
            model_name (str): Name of the model (used as the filename).
        """
        model_file = os.path.join(self.save_dir, f"{model_name}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved: {model_file}")
    
    def load_model(self, model_name):
        """
        Load a saved model from a file.
        
        Args:
            model_name (str): Name of the model (used as the filename).
        
        Returns:
            The loaded model object.
        """
        model_file = os.path.join(self.save_dir, f"{model_name}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found.")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded: {model_file}")
        return model
