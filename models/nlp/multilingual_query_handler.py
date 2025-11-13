"""
NLP Module for Multilingual Query Processing
Using BERT/IndicBERT for farmer query understanding and response generation
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
from typing import Dict, List, Optional
import json
import re


class MultilingualQueryHandler:
    """Handle multilingual farmer queries using transformer models"""
    
    def __init__(self, model_name: str = 'ai4bharat/indic-bert'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Language detection
        self.supported_languages = {
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
        
        # Query intent categories
        self.intent_categories = [
            'soil_health',
            'fertilizer_recommendation',
            'crop_selection',
            'pest_disease',
            'irrigation',
            'general_query',
        ]
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        
    def initialize_models(self):
        """Initialize transformer models"""
        print(f"Loading model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.intent_categories)
            )
            self.model.to(self.device)
            
            # Initialize QA pipeline for question answering
            self.qa_pipeline = pipeline(
                'question-answering',
                model='distilbert-base-cased-distilled-squad',
                device=0 if self.device == 'cuda' else -1
            )
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Using fallback text processing")
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple heuristic-based detection
        # In production, use a proper language detection library like langdetect
        
        # Check for Devanagari script (Hindi, Marathi)
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        # Check for Bengali script
        elif re.search(r'[\u0980-\u09FF]', text):
            return 'bn'
        # Check for Telugu script
        elif re.search(r'[\u0C00-\u0C7F]', text):
            return 'te'
        # Check for Tamil script
        elif re.search(r'[\u0B80-\u0BFF]', text):
            return 'ta'
        # Check for Gujarati script
        elif re.search(r'[\u0A80-\u0AFF]', text):
            return 'gu'
        # Check for Kannada script
        elif re.search(r'[\u0C80-\u0CFF]', text):
            return 'kn'
        # Check for Malayalam script
        elif re.search(r'[\u0D00-\u0D7F]', text):
            return 'ml'
        # Check for Gurmukhi script (Punjabi)
        elif re.search(r'[\u0A00-\u0A7F]', text):
            return 'pa'
        else:
            return 'en'
    
    def classify_intent(self, query: str) -> str:
        """Classify the intent of the query"""
        # Keyword-based intent classification (fallback)
        query_lower = query.lower()
        
        # Soil-related keywords
        soil_keywords = ['soil', 'मिट्टी', 'मृदा', 'ph', 'texture', 'type']
        # Fertilizer keywords
        fertilizer_keywords = ['fertilizer', 'खाद', 'उर्वरक', 'npk', 'urea', 'dap']
        # Crop keywords
        crop_keywords = ['crop', 'फसल', 'plant', 'grow', 'sow']
        # Pest keywords
        pest_keywords = ['pest', 'disease', 'कीट', 'रोग', 'insect']
        # Irrigation keywords
        irrigation_keywords = ['water', 'irrigation', 'सिंचाई', 'पानी']
        
        if any(keyword in query_lower for keyword in soil_keywords):
            return 'soil_health'
        elif any(keyword in query_lower for keyword in fertilizer_keywords):
            return 'fertilizer_recommendation'
        elif any(keyword in query_lower for keyword in crop_keywords):
            return 'crop_selection'
        elif any(keyword in query_lower for keyword in pest_keywords):
            return 'pest_disease'
        elif any(keyword in query_lower for keyword in irrigation_keywords):
            return 'irrigation'
        else:
            return 'general_query'
    
    def extract_entities(self, query: str, language: str) -> Dict:
        """Extract entities from query (crop name, soil type, etc.)"""
        entities = {
            'crop': None,
            'soil_type': None,
            'nutrients': [],
            'quantity': None,
            'area': None,
        }
        
        query_lower = query.lower()
        
        # Extract crop names
        crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'potato', 'tomato',
                'धान', 'गेहूं', 'मक्का', 'कपास', 'गन्ना', 'आलू', 'टमाटर']
        for crop in crops:
            if crop in query_lower:
                entities['crop'] = crop
                break
        
        # Extract soil types
        soil_types = ['sandy', 'loamy', 'clayey', 'silty', 'peaty', 'chalky',
                     'रेतीली', 'दोमट', 'चिकनी']
        for soil in soil_types:
            if soil in query_lower:
                entities['soil_type'] = soil
                break
        
        # Extract nutrients
        if 'nitrogen' in query_lower or 'n' in query_lower or 'नाइट्रोजन' in query_lower:
            entities['nutrients'].append('N')
        if 'phosphorus' in query_lower or 'p' in query_lower or 'फास्फोरस' in query_lower:
            entities['nutrients'].append('P')
        if 'potassium' in query_lower or 'k' in query_lower or 'पोटैशियम' in query_lower:
            entities['nutrients'].append('K')
        
        # Extract area (simple regex)
        area_match = re.search(r'(\d+\.?\d*)\s*(hectare|acre|ha|एकड़)', query_lower)
        if area_match:
            entities['area'] = float(area_match.group(1))
        
        return entities
    
    def process_query(self, query: str) -> Dict:
        """Process complete query and extract all information"""
        # Detect language
        language = self.detect_language(query)
        
        # Classify intent
        intent = self.classify_intent(query)
        
        # Extract entities
        entities = self.extract_entities(query, language)
        
        result = {
            'original_query': query,
            'language': language,
            'language_name': self.supported_languages.get(language, 'Unknown'),
            'intent': intent,
            'entities': entities,
        }
        
        return result
    
    def generate_response(self, query_info: Dict, recommendation_data: Dict, 
                         target_language: str = 'en') -> str:
        """Generate response in target language"""
        
        intent = query_info['intent']
        
        # Response templates
        if intent == 'soil_health':
            response = self._generate_soil_health_response(recommendation_data, target_language)
        elif intent == 'fertilizer_recommendation':
            response = self._generate_fertilizer_response(recommendation_data, target_language)
        elif intent == 'crop_selection':
            response = self._generate_crop_response(recommendation_data, target_language)
        else:
            response = self._generate_general_response(recommendation_data, target_language)
        
        return response
    
    def _generate_soil_health_response(self, data: Dict, language: str) -> str:
        """Generate soil health response"""
        
        templates = {
            'en': """
Soil Health Analysis:
- Soil Type: {soil_type}
- pH Level: {ph}
- Nitrogen: {nitrogen} kg/ha
- Phosphorus: {phosphorus} kg/ha
- Potassium: {potassium} kg/ha
- Organic Matter: {organic_matter}%

Recommendations:
{recommendations}
            """,
            'hi': """
मिट्टी स्वास्थ्य विश्लेषण:
- मिट्टी का प्रकार: {soil_type}
- pH स्तर: {ph}
- नाइट्रोजन: {nitrogen} किलो/हेक्टेयर
- फास्फोरस: {phosphorus} किलो/हेक्टेयर
- पोटैशियम: {potassium} किलो/हेक्टेयर
- जैविक पदार्थ: {organic_matter}%

सिफारिशें:
{recommendations}
            """
        }
        
        template = templates.get(language, templates['en'])
        
        return template.format(
            soil_type=data.get('soil_type', 'Unknown'),
            ph=data.get('pH', 'N/A'),
            nitrogen=data.get('Nitrogen', 'N/A'),
            phosphorus=data.get('Phosphorus', 'N/A'),
            potassium=data.get('Potassium', 'N/A'),
            organic_matter=data.get('Organic_Matter', 'N/A'),
            recommendations=data.get('advice', 'Follow soil test recommendations')
        )
    
    def _generate_fertilizer_response(self, data: Dict, language: str) -> str:
        """Generate fertilizer recommendation response"""
        
        templates = {
            'en': """
Fertilizer Recommendation for {crop}:

Nutrient Deficit:
- Nitrogen deficit: {n_deficit} kg/ha
- Phosphorus deficit: {p_deficit} kg/ha
- Potassium deficit: {k_deficit} kg/ha

Recommended Fertilizers:
{fertilizers}

Application Instructions:
{instructions}

Additional Tips:
{tips}
            """,
            'hi': """
{crop} के लिए उर्वरक सिफारिश:

पोषक तत्व की कमी:
- नाइट्रोजन की कमी: {n_deficit} किलो/हेक्टेयर
- फास्फोरस की कमी: {p_deficit} किलो/हेक्टेयर
- पोटैशियम की कमी: {k_deficit} किलो/हेक्टेयर

अनुशंसित उर्वरक:
{fertilizers}

प्रयोग निर्देश:
{instructions}

अतिरिक्त सुझाव:
{tips}
            """
        }
        
        template = templates.get(language, templates['en'])
        
        # Format fertilizer list
        fertilizer_list = data.get('fertilizer_recommendations', [])
        fertilizer_text = '\n'.join([
            f"- {f.get('fertilizer', '')}: {f.get('quantity', '')} ({f.get('application', '')})"
            for f in fertilizer_list
        ])
        
        return template.format(
            crop=data.get('crop', 'your crop'),
            n_deficit=data.get('nutrient_deficit', {}).get('N_deficit', 0),
            p_deficit=data.get('nutrient_deficit', {}).get('P_deficit', 0),
            k_deficit=data.get('nutrient_deficit', {}).get('K_deficit', 0),
            fertilizers=fertilizer_text,
            instructions='\n'.join(data.get('timing_advice', ['Follow package instructions'])),
            tips='\n'.join(data.get('additional_tips', ['Consult local agriculture officer']))
        )
    
    def _generate_crop_response(self, data: Dict, language: str) -> str:
        """Generate crop selection response"""
        
        templates = {
            'en': "For {soil_type} soil, recommended crops are: {crops}. These crops are well-suited for your soil conditions.",
            'hi': "{soil_type} मिट्टी के लिए, अनुशंसित फसलें हैं: {crops}। ये फसलें आपकी मिट्टी की स्थिति के लिए उपयुक्त हैं।"
        }
        
        template = templates.get(language, templates['en'])
        
        return template.format(
            soil_type=data.get('soil_type', 'your'),
            crops=', '.join(data.get('recommended_crops', ['wheat', 'rice']))
        )
    
    def _generate_general_response(self, data: Dict, language: str) -> str:
        """Generate general response"""
        
        templates = {
            'en': "Thank you for your query. Based on the analysis, we recommend following the soil health and fertilizer guidelines provided.",
            'hi': "आपके प्रश्न के लिए धन्यवाद। विश्लेषण के आधार पर, हम प्रदान की गई मिट्टी स्वास्थ्य और उर्वरक दिशानिर्देशों का पालन करने की सलाह देते हैं।"
        }
        
        return templates.get(language, templates['en'])
    
    def answer_faq(self, question: str, language: str = 'en') -> str:
        """Answer frequently asked questions"""
        
        faq_database = {
            'en': {
                'when to apply fertilizer': 'Apply fertilizers in split doses: 50% as basal dose before sowing, 25% at 30 days after planting, and 25% at 60 days after planting.',
                'how much water': 'Water requirements vary by crop and soil type. Generally, maintain 60-80% field capacity throughout the growing season.',
                'organic vs chemical': 'Both have advantages. Organic fertilizers improve soil health long-term, while chemical fertilizers provide quick nutrient availability. Best practice is to use a combination.',
            },
            'hi': {
                'उर्वरक कब डालें': 'उर्वरकों को विभाजित खुराक में डालें: बुवाई से पहले 50% मूल खुराक, रोपण के 30 दिन बाद 25%, और रोपण के 60 दिन बाद 25%।',
                'कितना पानी': 'पानी की आवश्यकता फसल और मिट्टी के प्रकार के अनुसार भिन्न होती है। आम तौर पर, बढ़ते मौसम में 60-80% क्षेत्र क्षमता बनाए रखें।',
            }
        }
        
        question_lower = question.lower()
        faq_dict = faq_database.get(language, faq_database['en'])
        
        # Find best matching FAQ
        for faq_q, faq_a in faq_dict.items():
            if faq_q in question_lower:
                return faq_a
        
        return "I don't have a specific answer to that question. Please consult with a local agricultural expert."


class VoiceResponseGenerator:
    """Generate voice responses for farmer queries"""
    
    def __init__(self):
        self.supported_languages = ['en', 'hi', 'mr', 'te', 'ta', 'bn']
    
    def text_to_speech(self, text: str, language: str = 'en', 
                      output_file: str = 'response.mp3') -> str:
        """Convert text to speech (placeholder for TTS integration)"""
        
        # In production, integrate with services like:
        # - Google Cloud Text-to-Speech
        # - Amazon Polly
        # - Indian language TTS services
        
        print(f"Generating speech for: {text[:50]}...")
        print(f"Language: {language}")
        print(f"Output file: {output_file}")
        
        return output_file
    
    def generate_voice_response(self, recommendation: Dict, language: str = 'en') -> str:
        """Generate complete voice response from recommendation data"""
        
        # Create simplified, voice-friendly text
        text = f"For {recommendation.get('crop', 'your crop')}, "
        text += f"apply {len(recommendation.get('fertilizer_recommendations', []))} types of fertilizers. "
        
        for fert in recommendation.get('fertilizer_recommendations', [])[:3]:
            text += f"{fert.get('fertilizer', '')}, {fert.get('quantity', '')}. "
        
        # Convert to speech
        output_file = self.text_to_speech(text, language)
        
        return output_file


if __name__ == "__main__":
    # Example usage
    print("Multilingual Query Handler initialized\n")
    
    handler = MultilingualQueryHandler()
    
    # Test queries
    test_queries = [
        "What fertilizer should I use for wheat in loamy soil?",
        "धान के लिए कौन सा उर्वरक उपयोग करें?",
        "My soil pH is 5.5, what should I do?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = handler.process_query(query)
        print(f"Language: {result['language_name']}")
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print("-" * 60)
