import unittest
import pandas as pd
from src.data_preprocessing import encode_categorical_variables

class TestDataPreprocessing(unittest.TestCase):
    
    def test_encode_categorical_variables(self):
        # Sample user data
        data = {
            'user_id': [1, 2, 3],
            'age': [25, 30, 22],
            'gender': ['M', 'F', 'M'],
            'occupation': ['student', 'engineer', 'artist'],
            'zip_code': ['12345', '23456', '34567']
        }
        users = pd.DataFrame(data)
        
        # Encode categorical variables
        users_encoded = encode_categorical_variables(users)
        
        # Check if 'gender' is encoded correctly
        self.assertTrue(set(users_encoded['gender']) == {0, 1})
        
        # Check if 'occupation' has been transformed into integers
        self.assertTrue(users_encoded['occupation'].dtype == 'int64')
        
        # Ensure no NaN values after encoding
        self.assertFalse(users_encoded.isnull().values.any())

if __name__ == '__main__':
    unittest.main()