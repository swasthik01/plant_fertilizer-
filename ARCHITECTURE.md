# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Web Browser  │  │ Mobile App   │  │  API Client  │          │
│  │  (HTML/JS)   │  │   (Future)   │  │   (Python)   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   NGINX/PROXY   │
                    │   (Port 80)     │
                    └────────┬────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────▼──────────┐              ┌──────────▼─────────┐
│   STATIC FILES     │              │   BACKEND API      │
│   (Frontend)       │              │   (FastAPI)        │
│  - index.html      │              │   Port 8000        │
│  - styles.css      │              └──────────┬─────────┘
│  - app.js          │                         │
└────────────────────┘                         │
                                    ┌──────────┴──────────┐
                                    │                     │
                         ┌──────────▼─────────┐  ┌───────▼────────┐
                         │  MODEL LAYER       │  │  DATA LAYER    │
                         │                    │  │                │
                         │  ┌──────────────┐  │  │  ┌──────────┐  │
                         │  │ Soil         │  │  │  │ Uploads  │  │
                         │  │ Detector     │  │  │  │ Directory│  │
                         │  │ (CNN)        │  │  │  └──────────┘  │
                         │  └──────────────┘  │  │                │
                         │                    │  │  ┌──────────┐  │
                         │  ┌──────────────┐  │  │  │ Models   │  │
                         │  │ Fertilizer   │  │  │  │ (.pth)   │  │
                         │  │ Recommender  │  │  │  └──────────┘  │
                         │  │ (XGBoost)    │  │  │                │
                         │  └──────────────┘  │  │  ┌──────────┐  │
                         │                    │  │  │ Data     │  │
                         │  ┌──────────────┐  │  │  │ Files    │  │
                         │  │ NLP Handler  │  │  │  └──────────┘  │
                         │  │ (BERT)       │  │  │                │
                         │  └──────────────┘  │  └────────────────┘
                         └────────────────────┘
```

## Component Interaction Flow

### Flow 1: Soil Detection Only

```
User Upload Image
      │
      ▼
Frontend (app.js)
      │
      ▼
POST /api/v1/detect-soil
      │
      ▼
Backend Endpoint (main.py)
      │
      ▼
SoilDetectorInference
      │
      ├─► Load Image
      │
      ├─► Preprocess (224x224)
      │
      ├─► CNN Forward Pass
      │
      ├─► Extract Features
      │
      └─► Calculate Confidence
      │
      ▼
JSON Response
      │
      ▼
Frontend Display
```

### Flow 2: Complete Recommendation Pipeline

```
User Input
  ├─► Soil Image
  ├─► Crop Selection
  └─► Soil Parameters (pH, NPK)
      │
      ▼
Frontend Form Submit
      │
      ▼
POST /api/v1/complete-recommendation
      │
      ▼
Backend Processing
      │
      ├─► STEP 1: Soil Detection
      │   │
      │   ├─► Load & Preprocess Image
      │   │
      │   ├─► CNN Inference
      │   │
      │   └─► Get Soil Type
      │
      ├─► STEP 2: Fertilizer Recommendation
      │   │
      │   ├─► Calculate Nutrient Deficit
      │   │
      │   ├─► XGBoost Prediction
      │   │
      │   ├─► Generate Recommendations
      │   │
      │   └─► Calculate Quantities
      │
      └─► STEP 3: Response Generation
          │
          ├─► Combine Results
          │
          └─► Format JSON
      │
      ▼
Complete Response
      │
      ▼
Frontend Display
  ├─► Soil Type Card
  ├─► Nutrient Deficit
  ├─► Fertilizer List
  └─► Application Tips
```

### Flow 3: Query Processing

```
User Query (Any Language)
      │
      ▼
POST /api/v1/process-query
      │
      ▼
MultilingualQueryHandler
      │
      ├─► Detect Language
      │
      ├─► Classify Intent
      │
      ├─► Extract Entities
      │   ├─► Crop Name
      │   ├─► Soil Type
      │   └─► Nutrients
      │
      └─► Generate Response
      │
      ▼
JSON Response
      │
      ▼
Display Analysis
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   DATA SOURCES                          │
├─────────────────────────────────────────────────────────┤
│  • Soil Images                                          │
│  • Soil Test Results (pH, NPK)                          │
│  • Crop Information                                      │
│  • Field Parameters                                      │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING                          │
├─────────────────────────────────────────────────────────┤
│  Image:                      Tabular:                    │
│  • Resize to 224x224        • Clean missing values      │
│  • Normalize                • Encode categories         │
│  • Augmentation             • Feature engineering       │
│  • Color extraction         • Scaling                   │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                 MODEL INFERENCE                          │
├─────────────────────────────────────────────────────────┤
│  CNN Model:                 ML Model:                    │
│  • Input: 224x224x3        • Input: 13 features         │
│  • Output: 6 classes       • Output: Fertilizer type    │
│  • Confidence score        • Probabilities              │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│            RECOMMENDATION ENGINE                         │
├─────────────────────────────────────────────────────────┤
│  • Nutrient Deficit Calculation                         │
│  • Fertilizer Selection                                  │
│  • Quantity Calculation                                  │
│  • Application Timing                                    │
│  • pH Management                                         │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              RESPONSE GENERATION                         │
├─────────────────────────────────────────────────────────┤
│  • Format Results                                        │
│  • Translate (if needed)                                │
│  • Add Context                                           │
│  • Generate Tips                                         │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│                  USER OUTPUT                             │
├─────────────────────────────────────────────────────────┤
│  • Soil Type & Confidence                               │
│  • Nutrient Analysis                                     │
│  • Fertilizer Recommendations                            │
│  • Application Instructions                              │
│  • Additional Tips                                       │
└─────────────────────────────────────────────────────────┘
```

## Model Architecture Details

### Soil Detection CNN

```
Input Image (224x224x3)
        │
        ▼
┌───────────────────┐
│  EfficientNet-B0  │
│  (Pretrained)     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Feature Maps     │
│  (1280 features)  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dropout (0.3)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dense (512)      │
│  + ReLU           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dropout (0.2)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dense (256)      │
│  + ReLU           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dropout (0.1)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Dense (6)        │
│  Soil Classes     │
└─────────┬─────────┘
          │
          ▼
     Softmax Output
```

### Fertilizer Recommendation Model

```
Input Features (13 dimensions)
  ├─► pH
  ├─► Nitrogen
  ├─► Phosphorus
  ├─► Potassium
  ├─► Organic Matter
  ├─► Moisture
  ├─► Temperature
  ├─► Soil Type (encoded)
  ├─► Crop (encoded)
  ├─► N/P Ratio
  ├─► N/K Ratio
  ├─► P/K Ratio
  └─► NPK Sum
        │
        ▼
┌─────────────────────┐
│  StandardScaler     │
│  (Normalize)        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  XGBoost Classifier │
│  - 200 estimators   │
│  - max_depth: 10    │
│  - learning_rate:0.1│
└─────────┬───────────┘
          │
          ▼
  Fertilizer Type
        │
        ▼
┌─────────────────────┐
│  Post-Processing    │
│  - Quantity calc    │
│  - Timing advice    │
│  - pH adjustment    │
└─────────────────────┘
```

## Deployment Architecture

### Docker Deployment

```
┌─────────────────────────────────────────────────┐
│              Docker Host                        │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  Container: agrismart-frontend            │ │
│  │  ┌─────────────────────────────────────┐  │ │
│  │  │  Nginx                              │  │ │
│  │  │  - Port: 80                         │  │ │
│  │  │  - Serves: HTML/CSS/JS              │  │ │
│  │  │  - Proxy: /api/ → backend:8000      │  │ │
│  │  └─────────────────────────────────────┘  │ │
│  └────────────────┬──────────────────────────┘ │
│                   │                            │
│  ┌────────────────▼──────────────────────────┐ │
│  │  Container: agrismart-backend             │ │
│  │  ┌─────────────────────────────────────┐  │ │
│  │  │  FastAPI + Uvicorn                  │  │ │
│  │  │  - Port: 8000                       │  │ │
│  │  │  - Models loaded in memory          │  │ │
│  │  │  - API endpoints                    │  │ │
│  │  └─────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  Volumes:                                 │ │
│  │  - ./models → /app/models                │ │
│  │  - ./uploads → /app/uploads              │ │
│  │  - ./data → /app/data                    │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Cloud Deployment (AWS)

```
┌────────────────────────────────────────────────────┐
│                  AWS Cloud                         │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  CloudFront (CDN)                            │ │
│  │  - Static content caching                    │ │
│  └─────────────┬────────────────────────────────┘ │
│                │                                  │
│  ┌─────────────▼────────────────────────────────┐ │
│  │  Application Load Balancer                   │ │
│  │  - SSL/TLS termination                       │ │
│  │  - Health checks                             │ │
│  └─────────────┬────────────────────────────────┘ │
│                │                                  │
│  ┌─────────────▼────────────────────────────────┐ │
│  │  ECS Cluster                                 │ │
│  │  ┌────────────────────────────────────────┐  │ │
│  │  │  ECS Service                           │  │ │
│  │  │  - Task Definition                     │  │ │
│  │  │  - Auto Scaling (1-10 instances)       │  │ │
│  │  │                                        │  │ │
│  │  │  ┌──────────────┐  ┌──────────────┐   │  │ │
│  │  │  │  Container 1 │  │  Container 2 │   │  │ │
│  │  │  │  (Backend)   │  │  (Backend)   │   │  │ │
│  │  │  └──────────────┘  └──────────────┘   │  │ │
│  │  └────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  ECR (Container Registry)                    │ │
│  │  - Docker images                             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  S3 Buckets                                  │ │
│  │  - Model files                               │ │
│  │  - Static assets                             │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │  CloudWatch                                  │ │
│  │  - Logs                                      │ │
│  │  - Metrics                                   │ │
│  │  - Alarms                                    │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

## Technology Stack Layers

```
┌─────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                     │
├─────────────────────────────────────────────────────┤
│  • HTML5 • CSS3 • JavaScript ES6                    │
│  • Bootstrap 5 • Font Awesome                       │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              APPLICATION LAYER                      │
├─────────────────────────────────────────────────────┤
│  • FastAPI • Uvicorn • Pydantic                     │
│  • Python 3.10+ • Async/Await                       │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              BUSINESS LOGIC LAYER                   │
├─────────────────────────────────────────────────────┤
│  • FertilizerAdvisor • SoilDetectorInference        │
│  • MultilingualQueryHandler                         │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              MODEL LAYER                            │
├─────────────────────────────────────────────────────┤
│  • PyTorch (CNN) • XGBoost (ML)                     │
│  • Transformers (NLP) • Scikit-learn                │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              DATA LAYER                             │
├─────────────────────────────────────────────────────┤
│  • Image Files • CSV Data                           │
│  • Model Checkpoints (.pth, .pkl)                   │
│  • Configuration Files                              │
└─────────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────────┐
│              INFRASTRUCTURE LAYER                   │
├─────────────────────────────────────────────────────┤
│  • Docker • Nginx • Cloud Services                  │
│  • Kubernetes (optional)                            │
└─────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────┐
│              SECURITY LAYERS                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Network Level:                                     │
│  ├─► HTTPS/SSL encryption                          │
│  ├─► CORS configuration                            │
│  └─► Rate limiting                                 │
│                                                     │
│  Application Level:                                 │
│  ├─► Input validation (Pydantic)                   │
│  ├─► File type validation                          │
│  ├─► Size limits (10MB)                            │
│  └─► Sanitization                                  │
│                                                     │
│  Authentication (Future):                           │
│  ├─► JWT tokens                                    │
│  ├─► OAuth 2.0                                     │
│  └─► API keys                                      │
│                                                     │
│  Data Protection:                                   │
│  ├─► No PII collection                             │
│  ├─► Temporary file cleanup                        │
│  └─► Secure model storage                          │
└─────────────────────────────────────────────────────┘
```

---

© 2025 AgriSmart - System Architecture Documentation
