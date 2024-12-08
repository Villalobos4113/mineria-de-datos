import pandas as pd
import os

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

def get_top_movies_by_cluster(movies_clustered, ratings, cluster, top_n=10):
    """
    Get top-rated movies within a specific cluster.
    
    Parameters:
        movies_clustered (DataFrame): Movies DataFrame with cluster labels.
        ratings (DataFrame): Ratings information.
        cluster (int): Cluster number.
        top_n (int): Number of top movies to retrieve.
        
    Returns:
        top_movies (DataFrame): Top-rated movies in the cluster.
    """
    # Get movies in the specified cluster
    cluster_movies = movies_clustered[movies_clustered['cluster'] == cluster]
    
    # Merge with ratings
    cluster_ratings = pd.merge(cluster_movies, ratings, on='movie_id')
    
    # Calculate average rating and number of ratings per movie
    movie_stats = cluster_ratings.groupby(['movie_id', 'movie_title']).agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    # Filter movies with sufficient ratings
    movie_stats = movie_stats[movie_stats['num_ratings'] >= 20]
    
    # Sort by average rating
    top_movies = movie_stats.sort_values(by='avg_rating', ascending=False).head(top_n)
    
    return top_movies

def recommend_movies_user_based(user_id, users_clustered, movies_clustered, ratings, top_n=10):
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
    user_cluster = users_clustered.loc[users_clustered['user_id'] == user_id, 'cluster'].values
    if len(user_cluster) == 0:
        raise ValueError(f"User ID {user_id} not found.")
    user_cluster = user_cluster[0]
    
    # Get movies in the user's cluster
    top_movies = get_top_movies_by_cluster(movies_clustered, ratings, cluster=user_cluster, top_n=top_n)
    
    # Get movies already rated by the user
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].tolist()
    
    # Exclude already rated movies
    recommendations = top_movies[~top_movies['movie_id'].isin(rated_movies)]
    
    return recommendations[['movie_title', 'avg_rating', 'num_ratings']]
