"""
Quick start script to test the complete system
Demonstrates the full pipeline: Soil Detection -> Fertilizer Recommendation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.fertilizer_prediction.fertilizer_recommender import FertilizerAdvisor
from models.nlp.multilingual_query_handler import MultilingualQueryHandler
import json


def test_fertilizer_advisor():
    """Test the fertilizer recommendation system"""
    
    print("=" * 60)
    print("TESTING FERTILIZER RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Initialize advisor
    advisor = FertilizerAdvisor()
    
    # Sample soil data
    soil_data = {
        "pH": 6.5,
        "Nitrogen": 45,
        "Phosphorus": 25,
        "Potassium": 35,
        "Organic_Matter": 2.5,
        "Moisture": 35,
    }
    
    print("\n📊 Soil Test Results:")
    for key, value in soil_data.items():
        print(f"  {key}: {value}")
    
    # Generate recommendation
    print("\n🌾 Generating recommendation for Wheat crop...")
    
    recommendation = advisor.generate_complete_recommendation(
        soil_data=soil_data,
        soil_type="Loamy",
        crop="Wheat",
        field_area=1.0,
        growth_stage="Vegetative",
        prefer_organic=False
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("RECOMMENDATION RESULTS")
    print("=" * 60)
    
    print(f"\n🌱 Crop: {recommendation['crop']}")
    print(f"🏞️  Soil Type: {recommendation['soil_type']}")
    print(f"📏 Field Area: {recommendation['field_area']}")
    print(f"🌿 Growth Stage: {recommendation['growth_stage']}")
    
    print("\n⚠️  NUTRIENT DEFICIT:")
    deficit = recommendation['nutrient_deficit']
    print(f"  Nitrogen deficit: {deficit.get('N_deficit', 0):.2f} kg/ha")
    print(f"  Phosphorus deficit: {deficit.get('P_deficit', 0):.2f} kg/ha")
    print(f"  Potassium deficit: {deficit.get('K_deficit', 0):.2f} kg/ha")
    
    print("\n💊 RECOMMENDED FERTILIZERS:")
    for i, fert in enumerate(recommendation['fertilizer_recommendations'], 1):
        print(f"\n  {i}. {fert['fertilizer']}")
        print(f"     Quantity: {fert['quantity']}")
        print(f"     Application: {fert['application']}")
    
    print("\n🌡️  pH MANAGEMENT:")
    print(f"  {recommendation['pH_recommendation']}")
    
    print("\n💡 SOIL-SPECIFIC ADVICE:")
    for advice in recommendation['soil_specific_advice']:
        print(f"  • {advice}")
    
    print("\n⭐ ADDITIONAL TIPS:")
    for tip in recommendation['additional_tips'][:3]:
        print(f"  • {tip}")
    
    print("\n" + "=" * 60)


def test_nlp_handler():
    """Test the NLP query processing"""
    
    print("\n" + "=" * 60)
    print("TESTING NLP QUERY HANDLER")
    print("=" * 60)
    
    # Initialize handler
    handler = MultilingualQueryHandler()
    
    # Test queries
    test_queries = [
        "What fertilizer should I use for wheat in loamy soil?",
        "How much nitrogen does rice need?",
        "धान के लिए कौन सा उर्वरक उपयोग करें?",
        "My soil pH is 5.5, what should I do?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = handler.process_query(query)
        
        print(f"   Language: {result['language_name']}")
        print(f"   Intent: {result['intent']}")
        print(f"   Detected Crop: {result['entities'].get('crop', 'Not detected')}")
        print(f"   Detected Soil: {result['entities'].get('soil_type', 'Not detected')}")
        if result['entities'].get('nutrients'):
            print(f"   Nutrients: {', '.join(result['entities']['nutrients'])}")
        print("   " + "-" * 50)


def test_complete_pipeline():
    """Test the complete pipeline simulation"""
    
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    print("\n🔬 Step 1: Soil Detection (Simulated)")
    print("   Analyzing soil image...")
    
    # Simulated soil detection result
    soil_detection = {
        "soil_type": "Loamy",
        "confidence": 0.87,
        "quality_rating": "Good"
    }
    
    print(f"   ✓ Detected Soil Type: {soil_detection['soil_type']}")
    print(f"   ✓ Confidence: {soil_detection['confidence']*100:.1f}%")
    print(f"   ✓ Quality: {soil_detection['quality_rating']}")
    
    print("\n💊 Step 2: Fertilizer Recommendation")
    print("   Generating recommendations...")
    
    advisor = FertilizerAdvisor()
    
    soil_data = {
        "pH": 6.8,
        "Nitrogen": 50,
        "Phosphorus": 28,
        "Potassium": 38,
    }
    
    recommendation = advisor.generate_complete_recommendation(
        soil_data=soil_data,
        soil_type=soil_detection['soil_type'],
        crop="Maize",
        field_area=2.0,
        growth_stage="Vegetative"
    )
    
    print(f"   ✓ Crop: {recommendation['crop']}")
    print(f"   ✓ Field Area: {recommendation['field_area']}")
    print(f"   ✓ Number of Recommendations: {len(recommendation['fertilizer_recommendations'])}")
    
    print("\n📊 SUMMARY:")
    print(f"   Soil Type: {soil_detection['soil_type']}")
    print(f"   Confidence: {soil_detection['confidence']*100:.1f}%")
    print(f"   Fertilizers Recommended: {len(recommendation['fertilizer_recommendations'])}")
    
    print("\n✅ Pipeline Test Complete!")


def main():
    """Run all tests"""
    
    print("\n" + "=" * 60)
    print("PLANT-SPECIFIC FERTILIZER & SOIL RECOMMENDATION SYSTEM")
    print("Quick Start Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Fertilizer Advisor
        test_fertilizer_advisor()
        
        # Test 2: NLP Handler
        test_nlp_handler()
        
        # Test 3: Complete Pipeline
        test_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60)
        
        print("\n📚 Next Steps:")
        print("  1. Start the backend: uvicorn backend.main:app --reload")
        print("  2. Open frontend: frontend/index.html")
        print("  3. Upload soil images and get recommendations!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("This is expected if dependencies are not installed.")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
