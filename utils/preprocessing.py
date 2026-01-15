"""
Data Preprocessing Module for Dengue Outbreak Prediction System
This module handles data loading, cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DengueDataPreprocessor:
    """
    Main preprocessor class for handling dengue and weather data
    """
    
    def __init__(self, dengue_path, weather_path):
        """
        Initialize preprocessor with dataset paths
        
        Args:
            dengue_path (str): Path to dengue cases CSV file
            weather_path (str): Path to weather data CSV file
        """
        self.dengue_path = dengue_path
        self.weather_path = weather_path
        self.merged_data = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load dengue and weather datasets from CSV files
        
        Returns:
            tuple: (dengue_df, weather_df)
        """
        try:
            dengue_df = pd.read_csv(self.dengue_path)
            weather_df = pd.read_csv(self.weather_path)
            
            # Convert Date column to datetime
            dengue_df['Date'] = pd.to_datetime(dengue_df['Date'])
            weather_df['Date'] = pd.to_datetime(weather_df['Date'])
            
            print(f"✓ Loaded {len(dengue_df)} dengue records")
            print(f"✓ Loaded {len(weather_df)} weather records")
            
            return dengue_df, weather_df
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def merge_datasets(self, dengue_df, weather_df):
        """
        Merge dengue and weather datasets on Date and Region
        
        Args:
            dengue_df (DataFrame): Dengue cases data
            weather_df (DataFrame): Weather data
            
        Returns:
            DataFrame: Merged dataset
        """
        merged = pd.merge(
            dengue_df, 
            weather_df, 
            on=['Date', 'Year', 'Month', 'Week', 'Region'],
            how='inner'
        )
        
        print(f"✓ Merged dataset: {len(merged)} records")
        return merged
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Cleaned dataframe
        """
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠ Found {missing.sum()} missing values")
            
            # Forward fill for time series data
            df = df.fillna(method='ffill')
            
            # Backward fill for remaining NaNs
            df = df.fillna(method='bfill')
            
            print("✓ Missing values handled")
        else:
            print("✓ No missing values found")
            
        return df
    
    def create_lag_features(self, df, lag_periods=[1, 2, 3]):
        """
        Create lag features for dengue cases (previous weeks/months)
        
        Args:
            df (DataFrame): Input dataframe
            lag_periods (list): List of lag periods to create
            
        Returns:
            DataFrame: Dataframe with lag features
        """
        df = df.sort_values('Date').reset_index(drop=True)
        
        for lag in lag_periods:
            df[f'Dengue_Cases_Lag_{lag}'] = df['Dengue_Cases'].shift(lag)
        
        # Fill initial NaN values with 0
        for lag in lag_periods:
            df[f'Dengue_Cases_Lag_{lag}'].fillna(0, inplace=True)
        
        print(f"✓ Created lag features: {lag_periods}")
        return df
    
    def create_seasonal_features(self, df):
        """
        Create seasonal and cyclical features
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Dataframe with seasonal features
        """
        # Monsoon season (June-September)
        df['Is_Monsoon'] = df['Month'].apply(lambda x: 1 if x in [6, 7, 8, 9] else 0)
        
        # Pre-monsoon (March-May)
        df['Is_PreMonsoon'] = df['Month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
        
        # Post-monsoon (October-November)
        df['Is_PostMonsoon'] = df['Month'].apply(lambda x: 1 if x in [10, 11] else 0)
        
        # Winter (December-February)
        df['Is_Winter'] = df['Month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        
        # Cyclical encoding for month
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        print("✓ Created seasonal features")
        return df
    
    def create_risk_levels(self, df):
        """
        Create risk level categories based on dengue case counts
        
        Risk Levels:
        - Low: < 100 cases
        - Medium: 100-400 cases
        - High: > 400 cases
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Dataframe with risk levels
        """
        def assign_risk(cases):
            if cases < 100:
                return 'Low'
            elif cases < 400:
                return 'Medium'
            else:
                return 'High'
        
        df['Risk_Level'] = df['Dengue_Cases'].apply(assign_risk)
        
        # Numeric encoding for risk levels
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Risk_Level_Numeric'] = df['Risk_Level'].map(risk_mapping)
        
        print("✓ Created risk level categories")
        print(f"   Low: {(df['Risk_Level'] == 'Low').sum()} records")
        print(f"   Medium: {(df['Risk_Level'] == 'Medium').sum()} records")
        print(f"   High: {(df['Risk_Level'] == 'High').sum()} records")
        
        return df
    
    def normalize_features(self, df, feature_columns):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df (DataFrame): Input dataframe
            feature_columns (list): List of columns to normalize
            
        Returns:
            DataFrame: Dataframe with normalized features
        """
        df_normalized = df.copy()
        
        # Fit and transform the features
        df_normalized[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        print(f"✓ Normalized {len(feature_columns)} features")
        return df_normalized
    
    def prepare_training_data(self, df, target_column='Dengue_Cases', test_size=0.2):
        """
        Prepare data for model training
        
        Args:
            df (DataFrame): Preprocessed dataframe
            target_column (str): Name of target column
            test_size (float): Proportion of test set
            
        Returns:
            tuple: X_train, X_test, y_train, y_test, feature_names
        """
        # Define feature columns (exclude non-feature columns)
        exclude_columns = [
            'Date', 'Region', 'Dengue_Cases', 'Risk_Level', 'Year'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Split data chronologically (no shuffling for time series)
        split_index = int(len(df) * (1 - test_size))
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def run_full_pipeline(self):
        """
        Execute complete preprocessing pipeline
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, processed_df, feature_names
        """
        print("\n" + "="*60)
        print("DENGUE DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        print("Step 1: Loading datasets...")
        dengue_df, weather_df = self.load_data()
        
        # Step 2: Merge datasets
        print("\nStep 2: Merging datasets...")
        merged_df = self.merge_datasets(dengue_df, weather_df)
        
        # Step 3: Handle missing values
        print("\nStep 3: Handling missing values...")
        merged_df = self.handle_missing_values(merged_df)
        
        # Step 4: Create lag features
        print("\nStep 4: Creating lag features...")
        merged_df = self.create_lag_features(merged_df)
        
        # Step 5: Create seasonal features
        print("\nStep 5: Creating seasonal features...")
        merged_df = self.create_seasonal_features(merged_df)
        
        # Step 6: Create risk levels
        print("\nStep 6: Creating risk level categories...")
        merged_df = self.create_risk_levels(merged_df)
        
        # Step 7: Prepare training data
        print("\nStep 7: Preparing training and test sets...")
        X_train, X_test, y_train, y_test, feature_names = self.prepare_training_data(merged_df)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60 + "\n")
        
        self.merged_data = merged_df
        
        return X_train, X_test, y_train, y_test, merged_df, feature_names


def get_feature_importance_names():
    """
    Get human-readable feature names for interpretation
    
    Returns:
        dict: Mapping of feature codes to readable names
    """
    return {
        'Month': 'Month of Year',
        'Week': 'Week of Year',
        'Temperature_C': 'Temperature (°C)',
        'Rainfall_mm': 'Rainfall (mm)',
        'Humidity_Percent': 'Humidity (%)',
        'Dengue_Cases_Lag_1': 'Previous Week Cases',
        'Dengue_Cases_Lag_2': 'Two Weeks Ago Cases',
        'Dengue_Cases_Lag_3': 'Three Weeks Ago Cases',
        'Is_Monsoon': 'Monsoon Season',
        'Is_PreMonsoon': 'Pre-Monsoon Season',
        'Is_PostMonsoon': 'Post-Monsoon Season',
        'Is_Winter': 'Winter Season',
        'Month_Sin': 'Month (Cyclical Sin)',
        'Month_Cos': 'Month (Cyclical Cos)'
    }


if __name__ == "__main__":
    # Example usage
    preprocessor = DengueDataPreprocessor(
        dengue_path='../data/dengue_cases.csv',
        weather_path='../data/weather_data.csv'
    )
    
    X_train, X_test, y_train, y_test, df, features = preprocessor.run_full_pipeline()
    
    print("\nDataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nFeature list:")
    print(features)
