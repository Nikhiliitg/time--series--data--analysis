import logging
import os
def setup_logging(log_file="app.log"):
    try:
        log_dir = "../logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, log_file)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
        logging.info("Logging setup complete.")
    except Exception as e:
        print(f"Logging setup failed: {e}")
