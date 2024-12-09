import os

from src.data_preprocessing import preprocess_data
from src.clustering import cluster_users_and_movies
from src.recommendation import recommend_movies_user_based, load_clustered_data
from src.recommendation import evaluate_recommendation_system

def main():
    # Define paths
    raw_data_path = os.path.join('data', 'raw')
    processed_data_path = os.path.join('data', 'processed')

    # Create processed data directory if it doesn't exist
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    # Run preprocessing
    preprocess_data(raw_data_path, processed_data_path)

    # Perform clustering
    cluster_users_and_movies(processed_data_path, user_k=5, movie_k=10)

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

    # Evaluate the recommendation system
    print("Evaluating the recommendation system...")
    mae, rmse = evaluate_recommendation_system(processed_data_path)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("\n")

if __name__ == "__main__":
    main()