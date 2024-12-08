import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.clustering import (
    prepare_user_features,
    perform_kmeans,
    add_user_clusters,
    prepare_movie_features,
    perform_kmeans_movie,
    add_movie_clusters
)

class TestClustering(unittest.TestCase):
    
    def setUp(self):
        # Sample user data
        self.user_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 22, 25, 33],
            'gender': [1, 0, 1, 1, 0],
            'occupation': [6, 16, 15, 6, 16],
            'zip_code': ['12345', '23456', '34567', '45678', '56789']
        })
        
        # Sample movie data
        self.movie_data = pd.DataFrame({
            'movie_id': [1, 2, 3, 4, 5],
            'movie_title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'unknown': [0, 0, 0, 0, 0],
            'Action': [1, 0, 0, 1, 0],
            'Adventure': [0, 0, 1, 1, 0],
            'Animation': [0, 0, 0, 0, 1],
            "Children's": [0, 1, 0, 0, 0],
            'Comedy': [1, 0, 0, 0, 1],
            'Crime': [0, 0, 0, 0, 0],
            'Documentary': [0, 0, 0, 0, 0],
            'Drama': [0, 1, 0, 1, 0],
            'Fantasy': [0, 0, 0, 0, 0],
            'Film-Noir': [0, 0, 0, 0, 0],
            'Horror': [0, 0, 0, 0, 0],
            'Musical': [0, 0, 0, 0, 0],
            'Mystery': [0, 0, 0, 0, 0],
            'Romance': [0, 0, 0, 0, 0],
            'Sci-Fi': [0, 0, 0, 0, 0],
            'Thriller': [0, 0, 0, 0, 0],
            'War': [0, 0, 0, 0, 0],
            'Western': [0, 0, 0, 0, 0]
        })
    
    def test_prepare_user_features(self):
        # Test user feature preparation
        data_scaled, user_ids = prepare_user_features(self.user_data)
        
        # Check if scaling was done correctly
        scaler = StandardScaler()
        expected_scaled = scaler.fit_transform(self.user_data[['age', 'gender', 'occupation']])
        np.testing.assert_array_almost_equal(data_scaled, expected_scaled)
        
        # Check if user_ids are returned correctly
        pd.testing.assert_series_equal(user_ids, self.user_data['user_id'])
    
    def test_perform_kmeans(self):
        # Test K-Means clustering
        data_scaled, _ = prepare_user_features(self.user_data)
        kmeans, labels = perform_kmeans(data_scaled, n_clusters=2)
        
        # Check if labels are of correct length
        self.assertEqual(len(labels), len(self.user_data))
        
        # Check if KMeans object has correct attributes
        self.assertTrue(hasattr(kmeans, 'labels_'))
        self.assertTrue(hasattr(kmeans, 'cluster_centers_'))
    
    def test_add_user_clusters(self):
        # Test adding cluster labels to users
        labels = [0, 1, 0, 0, 1]
        users_with_clusters = add_user_clusters(self.user_data, labels)
        
        # Check if 'cluster' column exists
        self.assertIn('cluster', users_with_clusters.columns)
        
        # Check if cluster labels are correct
        self.assertListEqual(users_with_clusters['cluster'].tolist(), labels)
    
    def test_prepare_movie_features(self):
        # Test movie feature preparation
        data_scaled, movie_ids = prepare_movie_features(self.movie_data)
        
        # Check if scaling was done correctly
        scaler = StandardScaler()
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        expected_scaled = scaler.fit_transform(self.movie_data[genre_columns])
        np.testing.assert_array_almost_equal(data_scaled, expected_scaled)
        
        # Check if movie_ids are returned correctly
        pd.testing.assert_series_equal(movie_ids, self.movie_data['movie_id'])
    
    def test_perform_kmeans_movie(self):
        # Test K-Means clustering for movies
        movie_features_scaled, _ = prepare_movie_features(self.movie_data)
        kmeans, labels = perform_kmeans_movie(movie_features_scaled, n_clusters=2)
        
        # Check if labels are of correct length
        self.assertEqual(len(labels), len(self.movie_data))
        
        # Check if KMeans object has correct attributes
        self.assertTrue(hasattr(kmeans, 'labels_'))
        self.assertTrue(hasattr(kmeans, 'cluster_centers_'))
    
    def test_add_movie_clusters(self):
        # Test adding cluster labels to movies
        labels = [0, 1, 0, 1, 0]
        movies_with_clusters = add_movie_clusters(self.movie_data, labels)
        
        # Check if 'cluster' column exists
        self.assertIn('cluster', movies_with_clusters.columns)
        
        # Check if cluster labels are correct
        self.assertListEqual(movies_with_clusters['cluster'].tolist(), labels)

if __name__ == '__main__':
    unittest.main()