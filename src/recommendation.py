import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def load_clustered_data(processed_data_path):
    """
    Load clustered users and movies data.

    Parameters:
        processed_data_path (str): Path to the processed data directory.

    Returns:
        users_clustered (DataFrame): Users DataFrame with cluster labels.
        movies_clustered (DataFrame): Movies DataFrame with cluster labels.
        ratings (DataFrame): Ratings information.
    """
    users_clustered = pd.read_csv(os.path.join(processed_data_path, 'users_clustered.csv'))
    movies_clustered = pd.read_csv(os.path.join(processed_data_path, 'movies_clustered.csv'))
    ratings = pd.read_csv(os.path.join(processed_data_path, 'ratings.csv'))

    return users_clustered, movies_clustered, ratings

def recommend_movies_user_based(user_id, users_clustered, movies_clustered, ratings, top_n=5):
    """
    Recommend top N movies to a user based on their cluster.

    Parameters:
        user_id (int): ID of the user.
        users_clustered (DataFrame): Users DataFrame with cluster labels.
        movies_clustered (DataFrame): Movies DataFrame with cluster labels.
        ratings (DataFrame): Ratings information.
        top_n (int): Number of recommendations.

    Returns:
        recommendations (DataFrame): Recommended movies with average ratings.
    """
    # Get user's cluster
    user_cluster = users_clustered.loc[users_clustered['user_id'] == user_id, 'user_cluster'].values
    if len(user_cluster) == 0:
        raise ValueError(f"User ID {user_id} not found.")
    user_cluster = user_cluster[0]

    # Get movies in the user's cluster
    cluster_movies = movies_clustered[movies_clustered['movie_cluster'] == user_cluster]

    # Merge with ratings
    cluster_ratings = pd.merge(cluster_movies, ratings, on='movie_id')

    # Calculate average rating and number of ratings per movie
    movie_stats = cluster_ratings.groupby(['movie_id', 'movie_title']).agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    # Filter movies with sufficient ratings (e.g., >= 20)
    movie_stats = movie_stats[movie_stats['num_ratings'] >= 20]

    # Sort by average rating
    top_movies = movie_stats.sort_values(by='avg_rating', ascending=False).head(top_n)

    # Get movies already rated by the user
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()

    # Exclude already rated movies
    recommendations = top_movies[~top_movies['movie_id'].isin(rated_movies)]

    return recommendations[['movie_title', 'avg_rating', 'num_ratings']]

def evaluate_recommendation_system(processed_data_path):
    """
    Evaluate the recommendation system using MAE and RMSE.

    Parameters:
        processed_data_path (str): Path to the processed data directory.

    Returns:
        mae (float): Mean Absolute Error.
        rmse (float): Root Mean Squared Error.
    """
    users_clustered, movies_clustered, ratings = load_clustered_data(processed_data_path)

    # Split into train and test sets (using existing splits or random split)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # Merge train set with movies and users to get cluster information
    # Specify suffixes to avoid column name conflicts
    train_with_movies = pd.merge(train, movies_clustered, on='movie_id', how='left', suffixes=('', '_movie'))
    train_with_users = pd.merge(train_with_movies, users_clustered, on='user_id', how='left', suffixes=('_movie', '_user'))

    # After merging, 'movie_cluster' and 'user_cluster' exist
    # Use 'user_cluster' for grouping
    cluster_movie_avg = train_with_users.groupby(['user_cluster', 'movie_id']).agg(
        avg_rating=('rating', 'mean')
    ).reset_index()

    # Merge test set with cluster information
    test_with_movies = pd.merge(test, movies_clustered, on='movie_id', how='left', suffixes=('', '_movie'))
    test_with_users = pd.merge(test_with_movies, users_clustered, on='user_id', how='left', suffixes=('_movie', '_user'))

    # Merge with cluster_movie_avg to get predicted ratings
    test_predictions = pd.merge(test_with_users, cluster_movie_avg, on=['user_cluster', 'movie_id'], how='left')

    # Fill missing predictions with overall average rating
    overall_avg = train['rating'].mean()
    # Refactored line to avoid FutureWarning
    test_predictions['avg_rating'] = test_predictions['avg_rating'].fillna(overall_avg)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(test_predictions['rating'], test_predictions['avg_rating'])
    rmse = np.sqrt(mean_squared_error(test_predictions['rating'], test_predictions['avg_rating']))

    return mae, rmse