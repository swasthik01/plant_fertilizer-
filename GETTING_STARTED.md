# Getting Started Guide

## Quick Start (5 Minutes)

### Windows Users

1. **Run the setup script:**
   ```cmd
   setup.bat
   ```

2. **Start the backend:**
   ```cmd
   venv\Scripts\activate
   uvicorn backend.main:app --reload
   ```

3. **Open the frontend:**
   - Open `frontend/index.html` in your web browser
   - Or visit `http://localhost:8000/docs` for API documentation

### Linux/Mac Users

1. **Make setup script executable and run:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Start the backend:**
   ```bash
   source venv/bin/activate
   uvicorn backend.main:app --reload
   ```

3. **Open the frontend:**
   - Open `frontend/index.html` in your web browser
   - Or visit `http://localhost:8000/docs` for API documentation

## Testing the System

### Option 1: Run Quick Test
```bash
python quick_start.py
```

This will test all components without requiring actual images or trained models.

### Option 2: Use the Web Interface

1. **Soil Detection:**
   - Navigate to the "Soil Detection" section
   - Upload a soil image
   - Get instant soil type detection results

2. **Complete Recommendation:**
   - Navigate to "Complete Recommendation System"
   - Upload a soil image
   - Fill in soil parameters (pH, NPK values)
   - Select your crop
   - Get personalized fertilizer recommendations

3. **Ask Questions:**
   - Navigate to "Ask a Question"
   - Type your query in English or any supported language
   - Get intent analysis and recommendations

## Using Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Access the application
Frontend: http://localhost:80
Backend API: http://localhost:8000
API Docs: http://localhost:8000/docs
```

## Training Your Own Models

### 1. Prepare Soil Images

Organize your soil images in this structure:
```
data/soil_images/
  Sandy/
    image1.jpg
    image2.jpg
  Loamy/
    image1.jpg
    image2.jpg
  Clayey/
    ...
  Silty/
    ...
  Peaty/
    ...
  Chalky/
    ...
```

### 2. Train Soil Detection Model

```bash
python notebooks/train_soil_detector.py
```

This will:
- Load and preprocess soil images
- Train EfficientNet model
- Save trained model to `models/soil_detection/best_model.pth`
- Generate training history plot

### 3. Train Fertilizer Model

```bash
python notebooks/train_fertilizer_model.py
```

This will:
- Generate/load soil and crop data
- Train XGBoost classifier
- Save model to `models/fertilizer_prediction/fertilizer_model.pkl`
- Display feature importance

## API Usage Examples

### Example 1: Detect Soil Type

```python
import requests

url = "http://localhost:8000/api/v1/detect-soil"
files = {"file": open("soil_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Soil Type: {result['soil_type']}")
print(f"Confidence: {result['confidence']}")
```

### Example 2: Get Fertilizer Recommendation

```python
import requests

url = "http://localhost:8000/api/v1/recommend-fertilizer"

data = {
    "soil_data": {
        "pH": 6.5,
        "Nitrogen": 45,
        "Phosphorus": 25,
        "Potassium": 35
    },
    "soil_type": "Loamy",
    "crop": "Wheat",
    "field_area": 1.0,
    "growth_stage": "Vegetative",
    "prefer_organic": false
}

response = requests.post(url, json=data)
recommendation = response.json()

print(recommendation['fertilizer_recommendations'])
```

### Example 3: Complete Pipeline

```python
import requests

url = "http://localhost:8000/api/v1/complete-recommendation"

files = {"file": open("soil_image.jpg", "rb")}
data = {
    "crop": "Wheat",
    "pH": 6.5,
    "nitrogen": 45,
    "phosphorus": 25,
    "potassium": 35,
    "field_area": 1.0
}

response = requests.post(url, files=files, data=data)
result = response.json()

print("Soil Detection:", result['soil_detection'])
print("Recommendations:", result['fertilizer_recommendation'])
```

## Troubleshooting

### Issue: Import Errors

**Solution:**
```bash
# Make sure virtual environment is activated
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Model Not Found

**Solution:**
The system works in demo mode without trained models. To use actual models:
1. Train models using the training scripts
2. Or download pre-trained models and place them in:
   - `models/soil_detection/best_model.pth`
   - `models/fertilizer_prediction/fertilizer_model.pkl`

### Issue: CORS Errors

**Solution:**
Make sure the backend is running when accessing the frontend. If using a different port, update `API_BASE_URL` in `frontend/app.js`.

### Issue: Port Already in Use

**Solution:**
```bash
# Use a different port
uvicorn backend.main:app --reload --port 8001
```

## Configuration

### Changing API Settings

Edit `config.py`:

```python
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
}
```

### Adding New Crops

Edit `config.py`:

```python
CROP_REQUIREMENTS = {
    'YourCrop': {
        'N': [min, max],
        'P': [min, max],
        'K': [min, max],
        'pH': [min, max]
    }
}
```

### Adding New Fertilizers

Edit `config.py`:

```python
FERTILIZERS = {
    'YourFertilizer': {'N': xx, 'P': yy, 'K': zz}
}
```

## Performance Optimization

### 1. Use GPU for Model Inference

```python
# Models automatically use GPU if available
# Check: torch.cuda.is_available()
```

### 2. Batch Processing

```python
# Process multiple images at once
results = detector.batch_predict(image_paths)
```

### 3. Model Caching

Models are loaded once and cached in memory for faster subsequent requests.

## Deployment Checklist

- [ ] Train and save models
- [ ] Set environment variables
- [ ] Configure CORS for production domain
- [ ] Set up SSL/HTTPS
- [ ] Configure database (if needed)
- [ ] Set up monitoring and logging
- [ ] Configure auto-scaling
- [ ] Set up backup strategy

## Support

- **Documentation:** See README.md
- **API Docs:** http://localhost:8000/docs
- **Issues:** Create an issue on GitHub
- **Testing:** Run `python quick_start.py`

## Next Steps

1. ✅ Complete setup
2. ✅ Test the system
3. 📊 Collect soil images and data
4. 🎯 Train models with your data
5. 🚀 Deploy to production
6. 📱 Build mobile app (optional)
7. 🌍 Scale to multiple regions

---

**Happy Farming with AI! 🌾🚜**
