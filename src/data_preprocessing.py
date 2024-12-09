import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def load_data(raw_data_path):
    """
    Load raw data files into pandas DataFrames.
    
    Parameters:
        raw_data_path (str): Path to the raw data directory.
        
    Returns:
        users (DataFrame): User information.
        movies (DataFrame): Movie information.
        ratings (DataFrame): Ratings information.
    """

    # Define column names
    u_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    item_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
                   'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    
    # Load data
    users = pd.read_csv(os.path.join(raw_data_path, 'u.user'), sep='|', names=u_columns, encoding='latin-1')
    movies = pd.read_csv(os.path.join(raw_data_path, 'u.item'), sep='|', names=item_columns, encoding='latin-1', usecols=range(24))
    ratings = pd.read_csv(os.path.join(raw_data_path, 'u.data'), sep='\t', names=rating_columns, encoding='latin-1')
    
    return users, movies, ratings

def check_missing_values(*dfs, names):
    """
    Check and report missing values in DataFrames.
    
    Parameters:
        *dfs: Variable number of DataFrames to check.
        names (list): List of names corresponding to each DataFrame.
    """

    for df, name in zip(dfs, names):
        missing = df.isnull().sum()
        print(f"Missing values in {name}:")
        print(missing)
        print("\n")

def handle_missing_values(users, movies, ratings):
    """
    Handle missing values in the DataFrames.
    
    Parameters:
        users (DataFrame): User information.
        movies (DataFrame): Movie information.
        ratings (DataFrame): Ratings information.
        
    Returns:
        users, movies, ratings (DataFrames): Cleaned DataFrames.
    """

    # For users and ratings, dropping rows with any missing values is acceptable
    users = users.dropna()
    ratings = ratings.dropna()
    
    # For movies, only drop rows where 'movie_id' or 'movie_title' is missing
    movies = movies.dropna(subset=['movie_id', 'movie_title'])
    
    return users, movies, ratings

def encode_categorical_variables(users):
    """
    Encode categorical variables in the users DataFrame.
    
    Parameters:
        users (DataFrame): User information.
        
    Returns:
        users (DataFrame): DataFrame with encoded categorical variables.
    """

    le = LabelEncoder()
    
    # Encode gender
    users['gender'] = le.fit_transform(users['gender'])
    
    # Encode occupation
    users['occupation'] = le.fit_transform(users['occupation'])
    
    return users

def select_features(users, movies):
    """
    Select relevant features for clustering.
    
    Parameters:
        users (DataFrame): User information.
        movies (DataFrame): Movie information.
        
    Returns:
        user_features (DataFrame): Features for user clustering.
        movie_features (DataFrame): Features for movie clustering.
    """
    
    # User-based features
    user_features = users.drop(['zip_code'], axis=1)
    
    # Movie-based features: retain 'movie_id' and 'movie_title' along with genres
    genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    # Retain 'movie_id' and 'movie_title'
    movie_features = movies[['movie_id', 'movie_title'] + genre_columns]
    
    return user_features, movie_features

def save_processed_data(user_features, movie_features, ratings, processed_data_path):
    """
    Save processed DataFrames to CSV files.
    
    Parameters:
        user_features (DataFrame): Features for user clustering.
        movie_features (DataFrame): Features for movie clustering.
        ratings (DataFrame): Ratings information.
        processed_data_path (str): Path to the processed data directory.
    """

    user_features.to_csv(os.path.join(processed_data_path, 'users.csv'), index=False)
    movie_features.to_csv(os.path.join(processed_data_path, 'movies.csv'), index=False)
    ratings.to_csv(os.path.join(processed_data_path, 'ratings.csv'), index=False)

def preprocess_data(raw_data_path, processed_data_path):
    """
    Complete data preprocessing pipeline.
    
    Parameters:
        raw_data_path (str): Path to the raw data directory.
        processed_data_path (str): Path to the processed data directory.
    """

    # Load data
    users, movies, ratings = load_data(raw_data_path)
    
    # Check for missing values
    check_missing_values(users, movies, ratings, names=['users', 'movies', 'ratings'])
    
    # Handle missing values
    users, movies, ratings = handle_missing_values(users, movies, ratings)
    
    # Encode categorical variables
    users = encode_categorical_variables(users)
    
    # Select features
    user_features, movie_features = select_features(users, movies)
    
    # Save processed data
    save_processed_data(user_features, movie_features, ratings, processed_data_path)
    
    print("Data preprocessing completed successfully!")

    # Verify processed data
    verify_processed_data(processed_data_path)

def verify_processed_data(processed_data_path):
    """
    Verify the processed data by loading and displaying basic information.
    
    Parameters:
        processed_data_path (str): Path to the processed data directory.
    """
    users = pd.read_csv(os.path.join(processed_data_path, 'users.csv'))
    movies = pd.read_csv(os.path.join(processed_data_path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(processed_data_path, 'ratings.csv'))
    
    print("Users DataFrame:")
    print(users.head())
    print("\nMovies DataFrame:")
    print(movies.head())
    print("\nRatings DataFrame:")
    print(ratings.head())