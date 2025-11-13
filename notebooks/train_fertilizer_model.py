"""
Training script for Fertilizer Recommendation Model
Train XGBoost model on soil and crop data
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from models.fertilizer_prediction.fertilizer_recommender import FertilizerRecommendationModel
from utils.data_preprocessing import SoilDataPreprocessor, create_sample_dataset


def prepare_fertilizer_data():
    """Prepare data for fertilizer recommendation training"""
    
    # Create or load dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Create target variable (fertilizer type based on NPK deficit)
    def assign_fertilizer(row):
        n_def = max(0, 80 - row['Nitrogen'])
        p_def = max(0, 40 - row['Phosphorus'])
        k_def = max(0, 40 - row['Potassium'])
        
        if n_def > 40 and p_def > 20 and k_def > 20:
            return 'NPK_Complex'
        elif n_def > 30:
            return 'Urea'
        elif p_def > 20:
            return 'DAP'
        elif k_def > 20:
            return 'MOP'
        else:
            return 'Balanced_NPK'
    
    df['Recommended_Fertilizer'] = df.apply(assign_fertilizer, axis=1)
    
    return df


def train_fertilizer_model(data_df):
    """Train the fertilizer recommendation model"""
    
    print("Preparing features...")
    
    # Initialize preprocessor
    preprocessor = SoilDataPreprocessor()
    
    # Create features
    data_df = preprocessor.prepare_features(data_df)
    
    # Encode categorical variables
    categorical_cols = ['Soil_Type', 'Crop']
    data_df = preprocessor.encode_categorical(data_df, categorical_cols)
    
    # Define feature columns
    feature_cols = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Matter', 
                   'Moisture', 'Temperature', 'Soil_Type', 'Crop',
                   'N_P_ratio', 'N_K_ratio', 'P_K_ratio', 'NPK_sum']
    
    X = data_df[feature_cols]
    y = data_df['Recommended_Fertilizer']
    
    # Split data
    X_train, X_val, y_train, y_val = preprocessor.prepare_train_test_split(
        pd.concat([X, y], axis=1),
        target_col='Recommended_Fertilizer',
        test_size=0.2
    )
    
    # Create and train model
    print("Training model...")
    model = FertilizerRecommendationModel(model_type='classification')
    model.create_model()
    
    # Train
    model.train(X_train, y_train, X_val, y_val)
    
    # Get feature importance
    print("\nFeature Importance:")
    importance = model.get_feature_importance()
    print(importance.head(10))
    
    # Save model
    model_path = 'models/fertilizer_prediction/fertilizer_model.pkl'
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model


if __name__ == "__main__":
    print("=== Fertilizer Recommendation Model Training ===\n")
    
    # Prepare data
    data = prepare_fertilizer_data()
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution:\n{data['Recommended_Fertilizer'].value_counts()}\n")
    
    # Train model
    model = train_fertilizer_model(data)
    
    print("\nTraining completed successfully!")
