"""
Data Preprocessing Utilities for Soil and Fertilizer Data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import albumentations as A
from typing import Tuple, Dict, List
import json


class SoilDataPreprocessor:
    """Preprocessor for soil health card data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_soil_data(self, filepath: str) -> pd.DataFrame:
        """Load soil health data from CSV"""
        df = pd.read_csv(filepath)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing values"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables"""
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features"""
        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        return df
    
    def create_npk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived NPK ratio features"""
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorus'] + 1e-6)
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
        df['P_K_ratio'] = df['Phosphorus'] / (df['Potassium'] + 1e-6)
        df['NPK_sum'] = df['Nitrogen'] + df['Phosphorus'] + df['Potassium']
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, target_col: str, 
                                 test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test


class SoilImagePreprocessor:
    """Preprocessor for soil images"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self):
        """Data augmentation for training"""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_val_transforms(self):
        """Transforms for validation/inference"""
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def preprocess_image(self, image: np.ndarray, training: bool = False) -> np.ndarray:
        """Preprocess single image"""
        if training:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        return transformed['image']
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features from soil image"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        features = {
            'mean_r': np.mean(image[:, :, 0]),
            'mean_g': np.mean(image[:, :, 1]),
            'mean_b': np.mean(image[:, :, 2]),
            'std_r': np.std(image[:, :, 0]),
            'std_g': np.std(image[:, :, 1]),
            'std_b': np.std(image[:, :, 2]),
            'mean_h': np.mean(hsv[:, :, 0]),
            'mean_s': np.mean(hsv[:, :, 1]),
            'mean_v': np.mean(hsv[:, :, 2]),
            'mean_l': np.mean(lab[:, :, 0]),
            'mean_a': np.mean(lab[:, :, 1]),
            'mean_b_lab': np.mean(lab[:, :, 2]),
        }
        
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM and other methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        features = {
            'gradient_mean': np.mean(np.abs(sobelx) + np.abs(sobely)),
            'gradient_std': np.std(np.abs(sobelx) + np.abs(sobely)),
            'laplacian_mean': np.mean(np.abs(laplacian)),
            'laplacian_std': np.std(np.abs(laplacian)),
            'texture_variance': np.var(gray),
        }
        
        return features
    
    def batch_preprocess(self, image_paths: List[str], training: bool = False) -> np.ndarray:
        """Preprocess batch of images"""
        processed_images = []
        for path in image_paths:
            image = self.load_image(path)
            processed = self.preprocess_image(image, training)
            processed_images.append(processed)
        
        return np.array(processed_images)


class FertilizerDataProcessor:
    """Process fertilizer recommendation data"""
    
    def __init__(self):
        self.crop_nutrient_map = self._load_crop_nutrient_requirements()
        
    def _load_crop_nutrient_requirements(self) -> Dict:
        """Load crop-specific nutrient requirements"""
        # This would be loaded from a JSON file in production
        return {
            'rice': {'N': [60, 80], 'P': [30, 40], 'K': [30, 40]},
            'wheat': {'N': [80, 100], 'P': [40, 50], 'K': [30, 40]},
            'maize': {'N': [100, 120], 'P': [50, 60], 'K': [40, 50]},
            'cotton': {'N': [100, 120], 'P': [40, 50], 'K': [40, 50]},
            'sugarcane': {'N': [200, 250], 'P': [60, 80], 'K': [80, 100]},
            'potato': {'N': [100, 150], 'P': [60, 80], 'K': [100, 120]},
            'tomato': {'N': [120, 150], 'P': [50, 70], 'K': [80, 100]},
        }
    
    def calculate_nutrient_deficit(self, soil_npk: Dict[str, float], 
                                   crop: str, growth_stage: str) -> Dict[str, float]:
        """Calculate nutrient deficit for specific crop and growth stage"""
        crop_requirements = self.crop_nutrient_map.get(crop.lower(), {})
        
        deficit = {
            'N_deficit': max(0, np.mean(crop_requirements.get('N', [0])) - soil_npk.get('N', 0)),
            'P_deficit': max(0, np.mean(crop_requirements.get('P', [0])) - soil_npk.get('P', 0)),
            'K_deficit': max(0, np.mean(crop_requirements.get('K', [0])) - soil_npk.get('K', 0)),
        }
        
        return deficit
    
    def recommend_fertilizer_type(self, npk_deficit: Dict[str, float]) -> List[str]:
        """Recommend fertilizer types based on deficits"""
        recommendations = []
        
        # Complex fertilizers
        if npk_deficit['N_deficit'] > 40 and npk_deficit['P_deficit'] > 20 and npk_deficit['K_deficit'] > 20:
            recommendations.append('NPK 20:20:20')
        elif npk_deficit['N_deficit'] > 50:
            recommendations.append('NPK 28:28:0')
        
        # Single nutrient fertilizers
        if npk_deficit['N_deficit'] > 30:
            recommendations.append('Urea (46% N)')
        if npk_deficit['P_deficit'] > 20:
            recommendations.append('DAP (18-46-0)')
        if npk_deficit['K_deficit'] > 20:
            recommendations.append('MOP (0-0-60)')
        
        # Organic alternatives
        if npk_deficit['N_deficit'] > 20:
            recommendations.append('Compost/FYM')
        
        return recommendations if recommendations else ['Balanced NPK 10:10:10']
    
    def calculate_fertilizer_quantity(self, deficit: Dict[str, float], 
                                     field_area: float = 1.0) -> Dict[str, float]:
        """Calculate fertilizer quantity in kg per hectare"""
        # Conversion factors based on nutrient content
        quantities = {
            'Urea': (deficit['N_deficit'] / 0.46) * field_area,
            'DAP': (deficit['P_deficit'] / 0.46) * field_area,
            'MOP': (deficit['K_deficit'] / 0.60) * field_area,
            'NPK_complex': ((deficit['N_deficit'] + deficit['P_deficit'] + deficit['K_deficit']) / 0.60) * field_area,
        }
        
        return {k: round(v, 2) for k, v in quantities.items()}


def create_sample_dataset():
    """Create sample dataset for testing"""
    np.random.seed(42)
    
    # Soil types
    soil_types = ['Sandy', 'Loamy', 'Clayey', 'Silty', 'Peaty', 'Chalky']
    crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Potato', 'Tomato']
    
    n_samples = 1000
    
    data = {
        'pH': np.random.uniform(4.5, 8.5, n_samples),
        'Nitrogen': np.random.uniform(10, 150, n_samples),
        'Phosphorus': np.random.uniform(5, 80, n_samples),
        'Potassium': np.random.uniform(10, 120, n_samples),
        'Organic_Matter': np.random.uniform(0.5, 5.0, n_samples),
        'Moisture': np.random.uniform(10, 60, n_samples),
        'Temperature': np.random.uniform(15, 35, n_samples),
        'Soil_Type': np.random.choice(soil_types, n_samples),
        'Crop': np.random.choice(crops, n_samples),
    }
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Test preprocessing
    print("Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Test soil data preprocessor
    preprocessor = SoilDataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    print("\nData cleaned successfully")
    
    # Test image preprocessor
    image_preprocessor = SoilImagePreprocessor()
    print("\nImage preprocessor initialized")
    
    print("\nPreprocessing utilities ready!")
