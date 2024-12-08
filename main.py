import os

from src.data_preprocessing import preprocess_data
from src.clustering import cluster_users_and_movies

# -------------------
# SETTINGS
# -------------------

# Define paths
preprocess_data_bool = False
raw_data_path = os.path.join('data', 'raw')
processed_data_path = os.path.join('data', 'processed')

# -------------------
# -------------------

def process_data():
    # Create processed data directory if it doesn't exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # Run preprocessing
    preprocess_data(raw_data_path, processed_data_path)

def main():
    # Perform clustering
    # Adjust 'user_k' and 'movie_k' as per your requirements or based on EDA
    cluster_users_and_movies(processed_data_path, user_k=5, movie_k=10)

if __name__ == "__main__":
    if preprocess_data_bool:
        process_data()
    main()