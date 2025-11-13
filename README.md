# Plant-Specific Fertilizer and Soil Recommendation System

## Overview

An AI-powered agricultural advisory system that provides personalized fertilizer recommendations and soil health analysis using Machine Learning and Natural Language Processing.

## Features

- **Image-based Soil Detection**: CNN models (EfficientNet/ResNet) to identify soil types from images
- **Fertilizer Recommendation**: ML-driven recommendations based on soil nutrients and crop requirements
- **Multilingual Support**: NLP module supporting 10+ Indian languages
- **Real-time Analysis**: FastAPI backend for instant recommendations
- **User-friendly Interface**: Responsive web/mobile UI built with Bootstrap

## System Architecture

```
┌─────────────────┐
│   User Input    │
│  (Soil Image +  │
│   Parameters)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model 1:       │
│  Soil Detection │
│  (EfficientNet) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model 2:       │
│  Fertilizer     │
│  Recommendation │
│  (XGBoost)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NLP Module:    │
│  Response       │
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Personalized   │
│  Recommendation │
└─────────────────┘
```

## Technologies Used

### Backend
- Python 3.10+
- FastAPI
- PyTorch (Deep Learning)
- XGBoost (ML)
- Transformers (NLP)

### Frontend
- HTML5, CSS3, JavaScript
- Bootstrap 5
- Font Awesome

### Deployment
- Docker & Docker Compose
- AWS (ECS, ECR)
- Google Cloud Platform (App Engine)
- Hugging Face Spaces

## Installation

### Prerequisites
- Python 3.10 or higher
- pip
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd minipp
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up directory structure**
The directories are already created, but ensure you have:
- `data/soil_images/` - for training soil images
- `models/` - for saved model files
- `uploads/` - for temporary uploaded files

## Usage

### 1. Running the Backend API

```bash
# From project root
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 2. Opening the Frontend

Simply open `frontend/index.html` in a web browser, or serve it using:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000
```

Then visit `http://localhost:3000`

### 3. Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Frontend: http://localhost:80
# Backend: http://localhost:8000
```

## API Endpoints

### Soil Detection
```
POST /api/v1/detect-soil
- Upload soil image
- Returns: soil type, confidence, properties
```

### Fertilizer Recommendation
```
POST /api/v1/recommend-fertilizer
- Input: soil data, crop, parameters
- Returns: fertilizer recommendations
```

### Complete Pipeline
```
POST /api/v1/complete-recommendation
- Upload image + soil parameters
- Returns: soil detection + fertilizer recommendation
```

### Query Processing
```
POST /api/v1/process-query
- Input: multilingual query
- Returns: detected intent and entities
```

## Training Models

### Train Soil Detection Model

```bash
python notebooks/train_soil_detector.py
```

**Requirements:**
- Organize soil images in `data/soil_images/` with subdirectories for each soil type
- Structure:
  ```
  data/soil_images/
    Sandy/
      image1.jpg
      image2.jpg
    Loamy/
      image1.jpg
    ...
  ```

### Train Fertilizer Recommendation Model

```bash
python notebooks/train_fertilizer_model.py
```

## Project Structure

```
minipp/
├── backend/
│   └── main.py                 # FastAPI application
├── frontend/
│   ├── index.html             # Main web interface
│   ├── styles.css             # Custom styling
│   └── app.js                 # Frontend logic
├── models/
│   ├── soil_detection/
│   │   └── soil_detector.py   # CNN models
│   ├── fertilizer_prediction/
│   │   └── fertilizer_recommender.py  # ML models
│   └── nlp/
│       └── multilingual_query_handler.py  # NLP module
├── utils/
│   └── data_preprocessing.py  # Data processing utilities
├── notebooks/
│   ├── train_soil_detector.py
│   └── train_fertilizer_model.py
├── deployment/
│   ├── aws_deploy.sh
│   ├── gcp_deploy.yaml
│   ├── huggingface_spaces.py
│   └── nginx.conf
├── data/
│   ├── raw/
│   ├── processed/
│   └── soil_images/
├── config.py                   # Configuration settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Configuration

Key configurations are in `config.py`:

- Soil types and properties
- Crop nutrient requirements
- Fertilizer compositions
- Model parameters
- API settings
- Language support

## Deployment

### AWS Deployment

```bash
cd deployment
# Edit aws_deploy.sh with your AWS account details
bash aws_deploy.sh
```

### Google Cloud Platform

```bash
gcloud app deploy deployment/gcp_deploy.yaml
```

### Hugging Face Spaces

```bash
# Upload deployment/huggingface_spaces.py to Hugging Face Spaces
# Set space type to "Gradio"
```

## Supported Languages

- English
- Hindi (हिन्दी)
- Marathi (मराठी)
- Telugu (తెలుగు)
- Tamil (தமிழ்)
- Bengali (বাংলা)
- Gujarati (ગુજરાતી)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Punjabi (ਪੰਜਾਬੀ)

## Supported Crops

Rice, Wheat, Maize, Cotton, Sugarcane, Potato, Tomato, Onion, Cabbage, Carrot

## Supported Soil Types

1. **Sandy** - Light, dry, low nutrients
2. **Loamy** - Ideal, balanced properties
3. **Clayey** - Heavy, water-retentive
4. **Silty** - Fertile, good drainage
5. **Peaty** - High organic matter
6. **Chalky** - Alkaline, free-draining

## Expected Outcomes

✅ Accurate soil type detection from images (>85% accuracy target)
✅ Automated two-stage prediction system
✅ Personalized fertilizer recommendations
✅ Multilingual query support
✅ Reduced fertilizer misuse
✅ Improved crop yield
✅ Sustainable farming practices

## Future Enhancements

- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time crop disease detection
- [ ] Weather-based recommendations
- [ ] IoT sensor integration
- [ ] Farmer community platform
- [ ] Market price prediction

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is developed for educational and agricultural advancement purposes.

## Contact & Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Contact: agrismart@example.com

## Acknowledgments

- Agricultural research data from soil health card databases
- Pre-trained models from PyTorch and Hugging Face
- Bootstrap and Font Awesome for UI components

---

**Powered by Machine Learning and NLP for Sustainable Agriculture** 🌱

© 2025 AgriSmart - Plant-Specific Fertilizer & Soil Recommendation System
