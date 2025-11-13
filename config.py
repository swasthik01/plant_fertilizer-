"""
Configuration file for the Plant-Specific Fertilizer and Soil Recommendation System
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
BACKEND_DIR = BASE_DIR / 'backend'
FRONTEND_DIR = BASE_DIR / 'frontend'

# Data directories
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
SOIL_IMAGES_DIR = DATA_DIR / 'soil_images'

# Model directories
SOIL_DETECTION_MODEL_DIR = MODEL_DIR / 'soil_detection'
FERTILIZER_MODEL_DIR = MODEL_DIR / 'fertilizer_prediction'
NLP_MODEL_DIR = MODEL_DIR / 'nlp'

# Model configurations
SOIL_DETECTION_CONFIG = {
    'model_name': 'efficientnet_b0',
    'image_size': (224, 224),
    'num_classes': 6,  # Sandy, Loamy, Clayey, Silty, Peaty, Chalky
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'pretrained': True,
}

FERTILIZER_MODEL_CONFIG = {
    'model_type': 'xgboost',
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'random_state': 42,
}

NLP_CONFIG = {
    'model_name': 'ai4bharat/indic-bert',
    'max_length': 512,
    'supported_languages': ['en', 'hi', 'mr', 'te', 'ta', 'bn', 'gu', 'kn', 'ml', 'pa'],
}

# Soil type labels
SOIL_TYPES = {
    0: 'Sandy',
    1: 'Loamy',
    2: 'Clayey',
    3: 'Silty',
    4: 'Peaty',
    5: 'Chalky',
}

# Soil type properties
SOIL_PROPERTIES = {
    'Sandy': {
        'description': 'Light, dry, warm, acidic, and low in nutrients',
        'water_retention': 'Low',
        'drainage': 'Excellent',
        'recommended_for': ['Carrots', 'Radish', 'Potatoes'],
    },
    'Loamy': {
        'description': 'Ideal soil with good drainage and nutrient retention',
        'water_retention': 'Good',
        'drainage': 'Good',
        'recommended_for': ['Most vegetables', 'Wheat', 'Cotton'],
    },
    'Clayey': {
        'description': 'Heavy, holds water, slow to warm in spring',
        'water_retention': 'High',
        'drainage': 'Poor',
        'recommended_for': ['Rice', 'Lettuce', 'Cabbage'],
    },
    'Silty': {
        'description': 'Light, moisture-retentive, fertile with good drainage',
        'water_retention': 'Good',
        'drainage': 'Moderate',
        'recommended_for': ['Most crops', 'Vegetables'],
    },
    'Peaty': {
        'description': 'High in organic matter, acidic, retains moisture',
        'water_retention': 'Very High',
        'drainage': 'Good',
        'recommended_for': ['Brassicas', 'Legumes'],
    },
    'Chalky': {
        'description': 'Alkaline, stony, free-draining',
        'water_retention': 'Low',
        'drainage': 'Excellent',
        'recommended_for': ['Spinach', 'Beets', 'Cabbage'],
    },
}

# Crop nutrient requirements (kg/hectare)
CROP_REQUIREMENTS = {
    'Rice': {'N': [60, 80], 'P': [30, 40], 'K': [30, 40], 'pH': [5.5, 6.5]},
    'Wheat': {'N': [80, 100], 'P': [40, 50], 'K': [30, 40], 'pH': [6.0, 7.0]},
    'Maize': {'N': [100, 120], 'P': [50, 60], 'K': [40, 50], 'pH': [5.5, 7.0]},
    'Cotton': {'N': [100, 120], 'P': [40, 50], 'K': [40, 50], 'pH': [6.0, 7.5]},
    'Sugarcane': {'N': [200, 250], 'P': [60, 80], 'K': [80, 100], 'pH': [6.0, 7.5]},
    'Potato': {'N': [100, 150], 'P': [60, 80], 'K': [100, 120], 'pH': [5.0, 6.0]},
    'Tomato': {'N': [120, 150], 'P': [50, 70], 'K': [80, 100], 'pH': [6.0, 7.0]},
    'Onion': {'N': [80, 100], 'P': [40, 50], 'K': [60, 80], 'pH': [6.0, 7.0]},
    'Cabbage': {'N': [100, 120], 'P': [50, 60], 'K': [80, 100], 'pH': [6.0, 7.5]},
    'Carrot': {'N': [60, 80], 'P': [30, 40], 'K': [60, 80], 'pH': [5.5, 7.0]},
}

# Fertilizer types and compositions
FERTILIZERS = {
    'Urea': {'N': 46, 'P': 0, 'K': 0},
    'DAP': {'N': 18, 'P': 46, 'K': 0},
    'MOP': {'N': 0, 'P': 0, 'K': 60},
    'NPK 10:26:26': {'N': 10, 'P': 26, 'K': 26},
    'NPK 12:32:16': {'N': 12, 'P': 32, 'K': 16},
    'NPK 20:20:20': {'N': 20, 'P': 20, 'K': 20},
    'NPK 19:19:19': {'N': 19, 'P': 19, 'K': 19},
    'NPK 28:28:0': {'N': 28, 'P': 28, 'K': 0},
    'Ammonium Sulphate': {'N': 21, 'P': 0, 'K': 0},
    'Single Super Phosphate': {'N': 0, 'P': 16, 'K': 0},
}

# Organic amendments
ORGANIC_FERTILIZERS = {
    'FYM': {'N': 0.5, 'P': 0.2, 'K': 0.5, 'application_rate': '10-15 tons/hectare'},
    'Compost': {'N': 0.8, 'P': 0.4, 'K': 0.8, 'application_rate': '5-10 tons/hectare'},
    'Vermicompost': {'N': 1.5, 'P': 1.0, 'K': 1.0, 'application_rate': '3-5 tons/hectare'},
    'Neem Cake': {'N': 5.0, 'P': 1.0, 'K': 1.4, 'application_rate': '200-400 kg/hectare'},
    'Poultry Manure': {'N': 3.0, 'P': 2.5, 'K': 1.5, 'application_rate': '2-4 tons/hectare'},
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'title': 'Plant-Specific Fertilizer & Soil Recommendation API',
    'version': '1.0.0',
    'description': 'AI-powered system for soil detection and fertilizer recommendations',
}

# File upload settings
UPLOAD_CONFIG = {
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
    'upload_folder': 'uploads',
}

# Language codes
LANGUAGE_CODES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
}

# Response templates
RESPONSE_TEMPLATES = {
    'en': {
        'soil_detected': 'Your soil type is {soil_type}.',
        'fertilizer_recommendation': 'For {crop}, we recommend applying {fertilizer} at {quantity} kg/hectare.',
        'deficit_message': 'Your soil has a deficit of: Nitrogen: {n} kg/ha, Phosphorus: {p} kg/ha, Potassium: {k} kg/ha.',
        'application_timing': 'Apply {percentage}% at {stage} stage.',
    },
    'hi': {
        'soil_detected': 'आपकी मिट्टी का प्रकार {soil_type} है।',
        'fertilizer_recommendation': '{crop} के लिए, हम {fertilizer} को {quantity} किलो/हेक्टेयर की दर से लगाने की सलाह देते हैं।',
        'deficit_message': 'आपकी मिट्टी में कमी है: नाइट्रोजन: {n} किलो/हेक्टेयर, फास्फोरस: {p} किलो/हेक्टेयर, पोटैशियम: {k} किलो/हेक्टेयर।',
        'application_timing': '{stage} अवस्था में {percentage}% डालें।',
    },
}

# Growth stages
GROWTH_STAGES = {
    'Vegetative': 'Early growth phase requiring high nitrogen',
    'Flowering': 'Flowering phase requiring balanced NPK with emphasis on P and K',
    'Fruiting': 'Fruit development requiring high potassium',
    'Ripening': 'Final stage requiring minimal nutrients',
}

# Database configuration (for future use)
DATABASE_CONFIG = {
    'db_type': 'sqlite',
    'db_name': 'fertilizer_recommendations.db',
    'tables': ['soil_tests', 'recommendations', 'user_feedback'],
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 SOIL_IMAGES_DIR, SOIL_DETECTION_MODEL_DIR, 
                 FERTILIZER_MODEL_DIR, NLP_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
