import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)

def load_processed_data(processed_data_path):
    """
    Load processed data files into pandas DataFrames.
    
    Parameters:
        processed_data_path (str): Path to the processed data directory.
        
    Returns:
        users (DataFrame): User information.
        movies (DataFrame): Movie information.
        ratings (DataFrame): Ratings information.
    """

    users = pd.read_csv(os.path.join(processed_data_path, 'users_clustered.csv'))
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies_clustered.csv'))
    ratings = pd.read_csv(os.path.join(processed_data_path, 'ratings.csv'))
    
    return users, movies, ratings

# -------------------
# User-Based Clustering
# -------------------

def prepare_user_features(users, ratings, movies):
    """
    Prepare user features for clustering, including rating statistics and genre taste vectors.
    The threshold for determining 'liked' movies is now dynamic:
    It is set to 75% of the user's maximum average rating across genres.
    
    Parameters:
        users (DataFrame): User information.
        ratings (DataFrame): Ratings information.
        movies (DataFrame): Movie information with genre columns.
        
    Returns:
        user_features_scaled (ndarray): Scaled user features.
        user_ids (Series): User IDs corresponding to the features.
    """

    # Calculate user rating statistics
    user_stats = ratings.groupby('user_id').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count'),
        rating_std=('rating', 'std')
    ).reset_index()
    user_stats['rating_std'].fillna(0, inplace=True)

    users_with_stats = pd.merge(users, user_stats, on='user_id', how='left')

    # Define genre columns
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                     'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                     'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    # Merge ratings with movies to get genre info for each rated movie
    ratings_with_genres = pd.merge(ratings, movies[['movie_id'] + genre_columns], on='movie_id', how='left')

    melted = ratings_with_genres.melt(
        id_vars=['user_id', 'movie_id', 'rating'],
        value_vars=genre_columns,
        var_name='genre',
        value_name='is_genre'
    )
    # Filter rows where the movie actually belongs to the genre
    melted = melted[melted['is_genre'] == 1]

    # Compute user-genre average ratings
    user_genre_avg = melted.groupby(['user_id', 'genre'])['rating'].mean().reset_index()

    # Find the maximum average genre rating per user
    user_max_genre = user_genre_avg.groupby('user_id')['rating'].max().reset_index()
    user_max_genre.rename(columns={'rating': 'max_genre_rating'}, inplace=True)

    # Merge the max genre rating back to user_stats
    users_with_threshold = pd.merge(users_with_stats, user_max_genre, on='user_id', how='left')
    # If a user has no genres rated, max_genre_rating might be NaN. Fill with global average or a default.
    global_avg = ratings['rating'].mean()
    users_with_threshold['max_genre_rating'].fillna(global_avg, inplace=True)

    # Threshold is 75% of max_genre_rating
    users_with_threshold['rating_threshold'] = users_with_threshold['max_genre_rating'] * 0.75

    # Merge threshold back to ratings to filter "liked" movies per user
    ratings_with_threshold = pd.merge(ratings_with_genres, users_with_threshold[['user_id', 'rating_threshold']], on='user_id', how='left')

    # Filter only liked movies
    liked_movies = ratings_with_threshold[ratings_with_threshold['rating'] > ratings_with_threshold['rating_threshold']]

    # Aggregate genre tastes based on liked movies
    user_genre_taste = liked_movies.groupby('user_id')[genre_columns].mean().reset_index().fillna(0)

    # Merge genre tastes with users_with_threshold
    users_with_taste = pd.merge(users_with_threshold, user_genre_taste, on='user_id', how='left')
    # Fill NaNs for users who did not have liked movies
    users_with_taste[genre_columns] = users_with_taste[genre_columns].fillna(0)

    # Select relevant features including genre tastes
    user_features = users_with_taste[['age', 'gender', 'occupation', 'avg_rating', 'rating_count', 'rating_std'] + genre_columns]

    user_features['gender'] = user_features['gender'].map({'M':0, 'F':1})
    
    # Handle any missing values
    user_features.fillna(0, inplace=True)

    # Scale the features
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)

    return user_features_scaled, users_with_taste['user_id']

def perform_kmeans(data, n_clusters):
    """
    Perform K-Means clustering.
    
    Parameters:
        data (ndarray): Scaled feature data.
        n_clusters (int): Number of clusters.
        
    Returns:
        kmeans (KMeans): Trained KMeans model.
        labels (ndarray): Cluster labels for each data point.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return kmeans, labels

def add_user_clusters(users, labels):
    """
    Add cluster labels to the users DataFrame.
    
    Parameters:
        users (DataFrame): User information.
        labels (ndarray): Cluster labels.
        
    Returns:
        users_with_clusters (DataFrame): Users DataFrame with cluster labels.
    """
    users_with_clusters = users.copy()
    users_with_clusters['user_cluster'] = labels
    return users_with_clusters

def visualize_user_clusters(data, labels, title='User Clusters'):
    """
    Visualize user clusters using PCA for dimensionality reduction.
    
    Parameters:
        data (ndarray): Scaled feature data.
        labels (ndarray): Cluster labels.
        title (str): Title of the plot.
        
    Returns:
        None
    """
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], hue=labels, palette='Set1')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()

# -------------------
# Movie-Based Clustering
# -------------------

def prepare_movie_features(movies):
    """
    Prepare movie features for clustering.
    
    Parameters:
        movies (DataFrame): Movie information.
        
    Returns:
        movie_features_scaled (ndarray): Scaled movie features.
        movie_ids (Series): Movie IDs corresponding to the features.
    """
    # Select genre columns
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movie_features = movies[genre_columns]
    
    # Scale the features
    scaler = StandardScaler()
    movie_features_scaled = scaler.fit_transform(movie_features)
    
    return movie_features_scaled, movies['movie_id']

def perform_kmeans_movie(data, n_clusters):
    """
    Perform K-Means clustering on movies.
    
    Parameters:
        data (ndarray): Scaled feature data.
        n_clusters (int): Number of clusters.
        
    Returns:
        kmeans (KMeans): Trained KMeans model.
        labels (ndarray): Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    return kmeans, labels

def add_movie_clusters(movies, labels):
    """
    Add cluster labels to the movies DataFrame.
    
    Parameters:
        movies (DataFrame): Movie information.
        labels (ndarray): Cluster labels.
        
    Returns:
        movies_with_clusters (DataFrame): Movies DataFrame with cluster labels.
    """
    movies_with_clusters = movies.copy()
    movies_with_clusters['movie_cluster'] = labels
    return movies_with_clusters

def visualize_movie_clusters(data, labels, title='Movie Clusters'):
    """
    Visualize movie clusters using PCA for dimensionality reduction.
    
    Parameters:
        data (ndarray): Scaled feature data.
        labels (ndarray): Cluster labels.
        title (str): Title of the plot.
        
    Returns:
        None
    """
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=principal_components[:,0], y=principal_components[:,1], hue=labels, palette='Set2')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.show()

# verify Clustered Data

def verify_clustered_data(processed_data_path):
    """
    Verify the clustered data by loading and displaying basic information.
    
    Parameters:
        processed_data_path (str): Path to the processed data directory.
    """
    users_clustered = pd.read_csv(os.path.join(processed_data_path, 'users_clustered.csv'))
    movies_clustered = pd.read_csv(os.path.join(processed_data_path, 'movies_clustered.csv'))
    
    print("Users with Clusters:")
    print(users_clustered.head())
    print("\nMovies with Clusters:")
    print(movies_clustered.head())

# Save Clustered Data

def save_clustered_data(users_with_clusters, movies_with_clusters, processed_data_path):
    """
    Save clustered users and movies to CSV files.
    
    Parameters:
        users_with_clusters (DataFrame): Users DataFrame with cluster labels.
        movies_with_clusters (DataFrame): Movies DataFrame with cluster labels.
        processed_data_path (str): Path to the processed data directory.
        
    Returns:
        None
    """
    users_with_clusters.to_csv(os.path.join(processed_data_path, 'users_clustered.csv'), index=False)
    movies_with_clusters.to_csv(os.path.join(processed_data_path, 'movies_clustered.csv'), index=False)
    print("Clustered data saved successfully!")

# Main Function

def cluster_users_and_movies(processed_data_path, user_k=5, movie_k=10):
    # Load data
    users, movies, ratings = load_processed_data(processed_data_path)

    # -------------------
    # User-Based Clustering
    # -------------------
    print("Starting User-Based Clustering...")

    # Prepare features with rating statistics and user taste profiles
    user_features_scaled, user_ids = prepare_user_features(users, ratings, movies)

    # Perform K-Means for users
    kmeans_users, user_labels = perform_kmeans(user_features_scaled, n_clusters=user_k)

    # Add cluster labels to users
    users_with_clusters = add_user_clusters(users, user_labels)

    # Visualize clusters
    visualize_user_clusters(user_features_scaled, user_labels, title='User Clusters')

    print(f"User-Based Clustering completed with {user_k} clusters.\n")

    # -------------------
    # Movie-Based Clustering
    # -------------------
    print("Starting Movie-Based Clustering...")

    movie_features_scaled, movie_ids = prepare_movie_features(movies)

    # Perform K-Means for movies
    kmeans_movies, movie_labels = perform_kmeans_movie(movie_features_scaled, n_clusters=movie_k)

    # Add cluster labels to movies
    movies_with_clusters = add_movie_clusters(movies, movie_labels)

    # Visualize clusters
    visualize_movie_clusters(movie_features_scaled, movie_labels, title='Movie Clusters')

    print(f"Movie-Based Clustering completed with {movie_k} clusters.\n")

    # Save Clustered Data
    save_clustered_data(users_with_clusters, movies_with_clusters, processed_data_path)

    # Verify clustered data
    verify_clustered_data(processed_data_path)