import os

from src.data_preprocessing import preprocess_data

def main():
    # Define paths
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')
    
    # Create processed data directory if it doesn't exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # Run preprocessing
    preprocess_data(raw_data_path, processed_data_path)

if __name__ == "__main__":
    main()