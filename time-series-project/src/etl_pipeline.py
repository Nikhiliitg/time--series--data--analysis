import boto3
import pandas as pd
import os

class ETLPipeline:
    def __init__(self, bucket_name, raw_file_name, local_raw_file, processed_dir):
        """
        Initialize the ETL pipeline class with the necessary parameters.
        """
        self.bucket_name = bucket_name
        self.raw_file_name = raw_file_name
        self.local_raw_file = local_raw_file
        self.processed_dir = processed_dir
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            print(f"Created processed directory: {self.processed_dir}")

        self.processed_file_name = os.path.join(self.processed_dir, 'processed_data.csv')

        self.s3 = boto3.client('s3', region_name='us-east-1')

    def extract_data_from_s3(self):
        """
        Extracts raw data from the S3 bucket and saves it locally.
        """
        try:
            self.s3.download_file(self.bucket_name, self.raw_file_name, self.local_raw_file)
            print(f"File {self.raw_file_name} downloaded successfully from S3.")
            data = pd.read_csv(self.local_raw_file)
            return data
        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return None

    def transform_data(self, data):
        data['Date'] = pd.to_datetime(data['Date'])
        
        data['Views'] = pd.to_numeric(data['Views'], errors='coerce')
        
        data.dropna(inplace=True)
        
        data['Day_of_Week'] = data['Date'].dt.day_name()
        
        print(f"Data transformed: {len(data)} rows after cleaning.")
        return data

    def load_data_locally(self, data):
        """
        Loads the processed data to a local file (CSV) in the processed directory.
        """
        try:
            data.to_csv(self.processed_file_name, index=False)
            print(f"Processed data saved locally as: {self.processed_file_name}")
        except Exception as e:
            print(f"Error saving file locally: {e}")

    def run_pipeline(self):
        """
        Orchestrates the entire ETL pipeline: extract, transform, and load.
        """
        raw_data = self.extract_data_from_s3()
        
        if raw_data is not None:
            transformed_data = self.transform_data(raw_data)
            
            self.load_data_locally(transformed_data)
        else:
            print("ETL pipeline failed during extraction. No data to process.")

if __name__ == "__main__":
    processed_dir = 'time-series-project/data/processed'
    local_raw_file = 'time-series-project/data/raw/final_data.csv'
    
    pipeline = ETLPipeline(
        bucket_name='myrawdata7',
        raw_file_name='final_data.csv',
        local_raw_file=local_raw_file,
        processed_dir=processed_dir
    )
    pipeline.run_pipeline()
