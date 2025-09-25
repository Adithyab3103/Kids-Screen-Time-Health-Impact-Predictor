# data_preprocessing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import joblib
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Handles data loading, cleaning, feature engineering, and preparation.
    Scaling is now handled within the model pipeline to prevent data leakage during cross-validation.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.mlb = MultiLabelBinarizer()
        self.feature_names = None
        self.target_names = None

    def load_data(self):
        """Loads the dataset from the specified path."""
        print(f"🔄 Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {self.data.shape}")
        return self.data

    def create_enhanced_features(self):
        """Creates additional features to improve model performance."""
        print("🔧 Creating enhanced features...")
        
        # Interaction term between age and screen time
        self.data['Age_Screen_Time_Interaction'] = self.data['Age'] * self.data['Avg_Daily_Screen_Time_hr']
        
        # Screen time risk categories
        self.data['Screen_Time_Risk'] = pd.cut(
            self.data['Avg_Daily_Screen_Time_hr'],
            bins=[0, 2, 4, 6, 8, float('inf')],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )

    def clean_data(self):
        """Cleans the dataset."""
        print("🧹 Cleaning data...")
        # Fill missing values if any (using median for numerical cols)
        for col in self.data.select_dtypes(include=np.number).columns:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].median(), inplace=True)
        # Convert boolean to integer
        self.data['Exceeded_Recommended_Limit'] = self.data['Exceeded_Recommended_Limit'].astype(int)

    def encode_features(self, X):
        """Encodes categorical features using one-hot encoding."""
        print("🔡 Encoding features...")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        self.feature_names = X_encoded.columns.tolist()
        return X_encoded

    def preprocess(self):
        """
        Main preprocessing pipeline that loads, cleans, engineers features,
        and prepares data for modeling.
        """
        self.load_data()
        self.create_enhanced_features()
        self.clean_data()

        # Handle the multi-label target
        self.data['Health_Impacts'] = self.data['Health_Impacts'].fillna('').astype(str)
        y_labels = self.data['Health_Impacts'].apply(lambda x: [label.strip() for label in x.split(',') if label.strip() and label.strip().lower() != 'none'])
        y = self.mlb.fit_transform(y_labels)
        self.target_names = self.mlb.classes_

        # Define features and drop the original target column
        X = self.data.drop('Health_Impacts', axis=1)
        X = self.encode_features(X)

        print("✅ Preprocessing complete.")
        return X, y

    def save_artifacts(self, save_dir='models'):
        """Saves the essential artifacts for inference (binarizer and feature names)."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.mlb, os.path.join(save_dir, 'multilabel_binarizer.joblib'))
        joblib.dump(self.feature_names, os.path.join(save_dir, 'feature_names.joblib'))
        print(f"✅ Binarizer and feature names saved to '{save_dir}'")