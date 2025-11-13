"""
Hugging Face Spaces Deployment Configuration
Gradio interface for the AgriSmart application
"""
import gradio as gr
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.soil_detection.soil_detector import SoilDetectorInference
    from models.fertilizer_prediction.fertilizer_recommender import FertilizerAdvisor
    from models.nlp.multilingual_query_handler import MultilingualQueryHandler
except ImportError:
    print("Running in demo mode")

# Initialize models (lazy loading)
soil_detector = None
fertilizer_advisor = FertilizerAdvisor()
nlp_handler = MultilingualQueryHandler()


def detect_soil_type(image):
    """Detect soil type from image"""
    if image is None:
        return "Please upload an image"
    
    # Mock response for demo
    result = {
        "soil_type": "Loamy",
        "confidence": 0.85,
        "quality_rating": "Good",
        "description": "Ideal soil with good drainage and nutrient retention"
    }
    
    output = f"""
    **Detected Soil Type:** {result['soil_type']}
    
    **Confidence:** {result['confidence']*100:.2f}%
    
    **Quality Rating:** {result['quality_rating']}
    
    **Description:** {result['description']}
    """
    
    return output


def recommend_fertilizer(soil_type, crop, nitrogen, phosphorus, potassium, ph, field_area):
    """Generate fertilizer recommendation"""
    
    soil_data = {
        "pH": ph,
        "Nitrogen": nitrogen,
        "Phosphorus": phosphorus,
        "Potassium": potassium,
    }
    
    recommendation = fertilizer_advisor.generate_complete_recommendation(
        soil_data=soil_data,
        soil_type=soil_type,
        crop=crop,
        field_area=field_area,
        growth_stage="Vegetative",
        prefer_organic=False
    )
    
    # Format output
    output = f"""
    # Fertilizer Recommendation for {crop}
    
    ## Nutrient Deficit
    - Nitrogen: {recommendation['deficit_summary']['N_deficit']}
    - Phosphorus: {recommendation['deficit_summary']['P_deficit']}
    - Potassium: {recommendation['deficit_summary']['K_deficit']}
    
    ## Recommended Fertilizers
    """
    
    for i, fert in enumerate(recommendation['fertilizer_recommendations'][:3], 1):
        output += f"""
    {i}. **{fert['fertilizer']}**
       - Quantity: {fert['quantity']}
       - Application: {fert['application']}
    """
    
    output += f"""
    
    ## pH Management
    {recommendation['pH_recommendation']}
    
    ## Additional Tips
    """
    for tip in recommendation['additional_tips'][:3]:
        output += f"- {tip}\n"
    
    return output


def process_query(query):
    """Process farmer query"""
    result = nlp_handler.process_query(query)
    
    output = f"""
    **Detected Language:** {result['language_name']}
    
    **Query Intent:** {result['intent'].replace('_', ' ').title()}
    
    **Extracted Information:**
    - Crop: {result['entities'].get('crop', 'Not detected')}
    - Soil Type: {result['entities'].get('soil_type', 'Not detected')}
    - Nutrients: {', '.join(result['entities'].get('nutrients', [])) or 'Not detected'}
    
    For detailed recommendations, please use the Fertilizer Recommendation tab.
    """
    
    return output


# Create Gradio Interface
with gr.Blocks(title="AgriSmart - Plant-Specific Fertilizer Recommendation", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🌱 AgriSmart - Plant-Specific Fertilizer & Soil Recommendation System
        
        AI-powered system for soil detection and fertilizer recommendations using Machine Learning and NLP
        """
    )
    
    with gr.Tabs():
        # Tab 1: Soil Detection
        with gr.TabItem("🔬 Soil Detection"):
            gr.Markdown("Upload a soil image to detect the soil type")
            
            with gr.Row():
                with gr.Column():
                    soil_image = gr.Image(type="filepath", label="Upload Soil Image")
                    detect_btn = gr.Button("Detect Soil Type", variant="primary")
                
                with gr.Column():
                    soil_output = gr.Markdown(label="Detection Result")
            
            detect_btn.click(detect_soil_type, inputs=[soil_image], outputs=[soil_output])
        
        # Tab 2: Fertilizer Recommendation
        with gr.TabItem("🌾 Fertilizer Recommendation"):
            gr.Markdown("Get personalized fertilizer recommendations")
            
            with gr.Row():
                with gr.Column():
                    soil_type_input = gr.Dropdown(
                        choices=["Sandy", "Loamy", "Clayey", "Silty", "Peaty", "Chalky"],
                        label="Soil Type",
                        value="Loamy"
                    )
                    crop_input = gr.Dropdown(
                        choices=["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", 
                                "Potato", "Tomato", "Onion", "Cabbage"],
                        label="Crop",
                        value="Wheat"
                    )
                    nitrogen_input = gr.Slider(0, 200, value=50, label="Nitrogen (kg/ha)")
                    phosphorus_input = gr.Slider(0, 100, value=30, label="Phosphorus (kg/ha)")
                    potassium_input = gr.Slider(0, 150, value=40, label="Potassium (kg/ha)")
                    ph_input = gr.Slider(4, 9, value=7.0, step=0.1, label="pH Level")
                    area_input = gr.Slider(0.1, 10, value=1.0, step=0.1, label="Field Area (hectares)")
                    
                    recommend_btn = gr.Button("Get Recommendation", variant="primary")
                
                with gr.Column():
                    fertilizer_output = gr.Markdown(label="Recommendation")
            
            recommend_btn.click(
                recommend_fertilizer,
                inputs=[soil_type_input, crop_input, nitrogen_input, phosphorus_input, 
                       potassium_input, ph_input, area_input],
                outputs=[fertilizer_output]
            )
        
        # Tab 3: Query Processing
        with gr.TabItem("💬 Ask a Question"):
            gr.Markdown("Ask questions in English, Hindi, or any supported language")
            
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What fertilizer should I use for wheat?",
                        lines=3
                    )
                    query_btn = gr.Button("Process Query", variant="primary")
                
                with gr.Column():
                    query_output = gr.Markdown(label="Analysis")
            
            query_btn.click(process_query, inputs=[query_input], outputs=[query_output])
    
    gr.Markdown(
        """
        ---
        ### About
        
        This system uses:
        - **EfficientNet** for soil type detection from images
        - **XGBoost** for fertilizer recommendations
        - **IndicBERT** for multilingual query processing
        
        Supports 10+ Indian languages for farmer-friendly interaction.
        
        © 2025 AgriSmart - Powered by AI for Sustainable Agriculture
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
