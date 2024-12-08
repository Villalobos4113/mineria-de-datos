import os

from src.data_preprocessing import preprocess_data
from src.clustering import cluster_users_and_movies
from src.recommendation import recommend_movies_user_based, load_clustered_data


# -------------------
# SETTINGS
# -------------------

# Define paths
raw_data_path = os.path.join('data', 'raw')
processed_data_path = os.path.join('data', 'processed')

# Processes
preprocess_data_bool = False
cluster = False

# -------------------
# -------------------

def process_data():
    # Create processed data directory if it doesn't exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
    
    # Run preprocessing
    preprocess_data(raw_data_path, processed_data_path)

def main():
    # Load clustered data
    users_clustered, movies_clustered, ratings = load_clustered_data(processed_data_path)
    
    # Interactive recommendation
    while True:
        try:
            user_input = input("Enter User ID to get recommendations (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                print("Exiting the recommendation system.")
                break
            user_id = int(user_input)
            recommendations = recommend_movies_user_based(user_id, users_clustered, movies_clustered, ratings, top_n=5)
            if recommendations.empty:
                print("No recommendations available. You might have rated all top movies in your cluster.")
            else:
                print(f"\nTop 5 movie recommendations for User ID {user_id}:\n")
                for idx, row in recommendations.iterrows():
                    print(f"{idx+1}. {row['movie_title']} - Average Rating: {row['avg_rating']:.2f} ({int(row['num_ratings'])} ratings)")
                print("\n")
        except ValueError as ve:
            print(str(ve))
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    

if __name__ == "__main__":
    if preprocess_data_bool:
        process_data()
    if cluster:
        # Adjust 'user_k' and 'movie_k' as per your requirements or based on EDA
        cluster_users_and_movies(processed_data_path, user_k=5, movie_k=10)
    main()