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

def prepare_user_features(users, ratings):
    """
    Prepare user features for clustering, including rating statistics.
    
    Parameters:
        users (DataFrame): User information.
        ratings (DataFrame): Ratings information.
        
    Returns:
        user_features_scaled (ndarray): Scaled user features.
        user_ids (Series): User IDs corresponding to the features.
    """

    # Add rating statistics
    user_stats = ratings.groupby('user_id').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count'),
        rating_std=('rating', 'std')
    ).reset_index()
    
    users_with_stats = pd.merge(users, user_stats, on='user_id', how='left')
    users_with_stats['rating_std'].fillna(0, inplace=True)  # Handle users with single rating
    
    # Select relevant features
    user_features = users_with_stats[['age', 'gender', 'occupation', 'avg_rating', 'rating_count', 'rating_std']]
    
    # Handle any missing values if necessary
    user_features.fillna(0, inplace=True)
    
    # Scale the features
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    return user_features_scaled, users_with_stats['user_id']

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
    """
    Perform clustering on users and movies.
    
    Parameters:
        processed_data_path (str): Path to the processed data directory.
        user_k (int): Number of clusters for users.
        movie_k (int): Number of clusters for movies.
        
    Returns:
        None
    """
    # Load data
    users, movies, ratings = load_processed_data(processed_data_path)
    
    # -------------------
    # User-Based Clustering
    # -------------------
    print("Starting User-Based Clustering...")
    
    # Prepare features with rating statistics
    user_features_scaled, user_ids = prepare_user_features(users, ratings)
    
    # Perform K-Means
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
    
    # Prepare features
    movie_features_scaled, movie_ids = prepare_movie_features(movies)
    
    # Perform K-Means
    kmeans_movies, movie_labels = perform_kmeans_movie(movie_features_scaled, n_clusters=movie_k)
    
    # Add cluster labels to movies
    movies_with_clusters = add_movie_clusters(movies, movie_labels)
    
    # Visualize clusters
    visualize_movie_clusters(movie_features_scaled, movie_labels, title='Movie Clusters')
    
    print(f"Movie-Based Clustering completed with {movie_k} clusters.\n")
    
    # -------------------
    # Save Clustered Data
    # -------------------
    save_clustered_data(users_with_clusters, movies_with_clusters, processed_data_path)
    
    # Verify clustered data
    verify_clustered_data(processed_data_path)