import pandas as pd
from io import StringIO

class DataIngestion:
    def __init__(self, file_storage):
        self.file_storage = file_storage

    def load_data(self):
        # Check the file extension
        filename = self.file_storage.filename
        if filename.endswith('.csv'):
            # Read CSV file
            data = pd.read_csv(self.file_storage)
        elif filename.endswith('.json'):
            # Read JSON file
            data = pd.read_json(self.file_storage)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")
        
        # Clean and preprocess the data
        data = self.preprocess_data(data)
        return data

    def preprocess_data(self, data):
        # Handle missing values
        data.fillna(0, inplace=True)

        # Convert categorical columns to numeric (if any)
        for col in data.columns:
            if data[col].dtype == 'object':  # Check if the column is non-numeric
                data[col] = pd.factorize(data[col])[0]  # Convert to numeric codes

        return data