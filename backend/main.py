"""
FastAPI Backend for Plant-Specific Fertilizer and Soil Recommendation System
Main API endpoints for soil detection and fertilizer recommendation
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import sys
from pathlib import Path
import shutil
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
try:
    from models.soil_detection.soil_detector import SoilDetectorInference
    from models.fertilizer_prediction.fertilizer_recommender import FertilizerAdvisor
    from models.nlp.multilingual_query_handler import MultilingualQueryHandler
except ImportError:
    print("Warning: Some modules could not be imported. Running in demo mode.")

# Initialize FastAPI app
app = FastAPI(
    title="Plant-Specific Fertilizer & Soil Recommendation API",
    description="AI-powered system for soil detection and fertilizer recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic models for request/response
class SoilData(BaseModel):
    pH: float
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    Organic_Matter: Optional[float] = 2.0
    Moisture: Optional[float] = 30.0
    Temperature: Optional[float] = 25.0

class FertilizerRequest(BaseModel):
    soil_data: Dict[str, float]
    soil_type: str
    crop: str
    field_area: Optional[float] = 1.0
    growth_stage: Optional[str] = "Vegetative"
    prefer_organic: Optional[bool] = False

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = "auto"

class SoilDetectionResponse(BaseModel):
    soil_type: str
    confidence: float
    all_probabilities: Dict[str, float]
    color_features: Dict[str, float]
    texture_features: Dict[str, float]
    quality_rating: str

class FertilizerResponse(BaseModel):
    crop: str
    soil_type: str
    field_area: str
    nutrient_deficit: Dict
    fertilizer_recommendations: List[Dict]
    deficit_summary: Dict
    pH_recommendation: str
    soil_specific_advice: List[str]
    additional_tips: List[str]

# Global model instances (lazy loading)
soil_detector = None
fertilizer_advisor = None
nlp_handler = None


def get_soil_detector():
    """Get or initialize soil detector"""
    global soil_detector
    if soil_detector is None:
        try:
            model_path = "models/soil_detection/best_model.pth"
            if os.path.exists(model_path):
                soil_detector = SoilDetectorInference(model_path)
            else:
                print("Soil detector model not found. Using demo mode.")
        except Exception as e:
            print(f"Error loading soil detector: {e}")
    return soil_detector


def get_fertilizer_advisor():
    """Get or initialize fertilizer advisor"""
    global fertilizer_advisor
    if fertilizer_advisor is None:
        try:
            fertilizer_advisor = FertilizerAdvisor()
        except Exception as e:
            print(f"Error initializing fertilizer advisor: {e}")
    return fertilizer_advisor


def get_nlp_handler():
    """Get or initialize NLP handler"""
    global nlp_handler
    if nlp_handler is None:
        try:
            nlp_handler = MultilingualQueryHandler()
        except Exception as e:
            print(f"Error initializing NLP handler: {e}")
    return nlp_handler


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant-Specific Fertilizer & Soil Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "soil_detection": "/api/v1/detect-soil",
            "fertilizer_recommendation": "/api/v1/recommend-fertilizer",
            "complete_recommendation": "/api/v1/complete-recommendation",
            "query_processing": "/api/v1/process-query",
            "health_check": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models": {
            "soil_detector": soil_detector is not None,
            "fertilizer_advisor": fertilizer_advisor is not None,
            "nlp_handler": nlp_handler is not None,
        }
    }


@app.post("/api/v1/detect-soil")
async def detect_soil(file: UploadFile = File(...)):
    """
    Detect soil type from uploaded image
    Model 1: Image-based soil detection
    """
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get soil detector
        detector = get_soil_detector()
        
        if detector is None:
            # Demo mode - return mock response
            result = {
                "soil_type": "Loamy",
                "confidence": 0.85,
                "all_probabilities": {
                    "Sandy": 0.05,
                    "Loamy": 0.85,
                    "Clayey": 0.05,
                    "Silty": 0.03,
                    "Peaty": 0.01,
                    "Chalky": 0.01
                },
                "color_features": {
                    "mean_r": 120.5,
                    "mean_g": 100.3,
                    "mean_b": 80.2
                },
                "texture_features": {
                    "variance": 850.5,
                    "roughness": 45.3
                },
                "quality_rating": "Good"
            }
        else:
            # Use actual model
            result = detector.predict(str(file_path))
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/recommend-fertilizer")
async def recommend_fertilizer(request: FertilizerRequest):
    """
    Recommend fertilizers based on soil data and crop
    Model 2: Fertilizer recommendation
    """
    try:
        # Get fertilizer advisor
        advisor = get_fertilizer_advisor()
        
        if advisor is None:
            raise HTTPException(status_code=500, detail="Fertilizer advisor not available")
        
        # Generate recommendation
        recommendation = advisor.generate_complete_recommendation(
            soil_data=request.soil_data,
            soil_type=request.soil_type,
            crop=request.crop,
            field_area=request.field_area,
            growth_stage=request.growth_stage,
            prefer_organic=request.prefer_organic
        )
        
        return JSONResponse(content=recommendation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/complete-recommendation")
async def complete_recommendation(
    file: UploadFile = File(...),
    crop: str = Form(...),
    pH: float = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    organic_matter: float = Form(2.0),
    moisture: float = Form(30.0),
    field_area: float = Form(1.0),
    growth_stage: str = Form("Vegetative"),
    prefer_organic: bool = Form(False)
):
    """
    Complete pipeline: Soil detection -> Fertilizer recommendation
    Sequential execution of Model 1 and Model 2
    """
    try:
        # Step 1: Detect soil type from image
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        detector = get_soil_detector()
        
        if detector is None:
            soil_result = {
                "soil_type": "Loamy",
                "confidence": 0.85,
                "quality_rating": "Good"
            }
        else:
            soil_result = detector.predict(str(file_path))
        
        os.remove(file_path)
        
        # Step 2: Generate fertilizer recommendation using detected soil type
        soil_data = {
            "pH": pH,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Organic_Matter": organic_matter,
            "Moisture": moisture
        }
        
        advisor = get_fertilizer_advisor()
        
        if advisor is None:
            raise HTTPException(status_code=500, detail="Fertilizer advisor not available")
        
        fertilizer_recommendation = advisor.generate_complete_recommendation(
            soil_data=soil_data,
            soil_type=soil_result['soil_type'],
            crop=crop,
            field_area=field_area,
            growth_stage=growth_stage,
            prefer_organic=prefer_organic
        )
        
        # Combine results
        complete_result = {
            "soil_detection": soil_result,
            "fertilizer_recommendation": fertilizer_recommendation,
            "pipeline_status": "success"
        }
        
        return JSONResponse(content=complete_result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/process-query")
async def process_query(request: QueryRequest):
    """
    Process multilingual farmer query
    NLP module for query understanding
    """
    try:
        handler = get_nlp_handler()
        
        if handler is None:
            raise HTTPException(status_code=500, detail="NLP handler not available")
        
        # Process query
        result = handler.process_query(request.query)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-response")
async def generate_response(
    query_info: Dict,
    recommendation_data: Dict,
    target_language: str = "en"
):
    """
    Generate natural language response in target language
    """
    try:
        handler = get_nlp_handler()
        
        if handler is None:
            raise HTTPException(status_code=500, detail="NLP handler not available")
        
        response = handler.generate_response(query_info, recommendation_data, target_language)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/crops")
async def get_crops():
    """Get list of supported crops"""
    crops = [
        "Rice", "Wheat", "Maize", "Cotton", "Sugarcane", 
        "Potato", "Tomato", "Onion", "Cabbage", "Carrot"
    ]
    return {"crops": crops}


@app.get("/api/v1/soil-types")
async def get_soil_types():
    """Get list of soil types"""
    soil_types = {
        "Sandy": "Light, dry, warm, acidic, and low in nutrients",
        "Loamy": "Ideal soil with good drainage and nutrient retention",
        "Clayey": "Heavy, holds water, slow to warm in spring",
        "Silty": "Light, moisture-retentive, fertile with good drainage",
        "Peaty": "High in organic matter, acidic, retains moisture",
        "Chalky": "Alkaline, stony, free-draining"
    }
    return {"soil_types": soil_types}


@app.get("/api/v1/fertilizers")
async def get_fertilizers():
    """Get list of available fertilizers"""
    fertilizers = {
        "Urea": {"N": 46, "P": 0, "K": 0},
        "DAP": {"N": 18, "P": 46, "K": 0},
        "MOP": {"N": 0, "P": 0, "K": 60},
        "NPK 20:20:20": {"N": 20, "P": 20, "K": 20},
    }
    return {"fertilizers": fertilizers}


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
