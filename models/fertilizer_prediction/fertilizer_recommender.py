"""
ML-based Fertilizer Recommendation System
Model 2: Plant-specific fertilizer prediction using XGBoost and Random Forest
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json
from typing import Dict, List, Tuple
from pathlib import Path


class FertilizerRecommendationModel:
    """XGBoost-based fertilizer recommendation model"""
    
    def __init__(self, model_type='classification'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
    def create_model(self, params=None):
        """Create XGBoost model"""
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
            }
        
        if self.model_type == 'classification':
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)
        
        return self.model
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        # Create derived features
        if 'Nitrogen' in data.columns and 'Phosphorus' in data.columns and 'Potassium' in data.columns:
            data['N_P_ratio'] = data['Nitrogen'] / (data['Phosphorus'] + 1e-6)
            data['N_K_ratio'] = data['Nitrogen'] / (data['Potassium'] + 1e-6)
            data['P_K_ratio'] = data['Phosphorus'] / (data['Potassium'] + 1e-6)
            data['NPK_sum'] = data['Nitrogen'] + data['Phosphorus'] + data['Potassium']
            data['NPK_balance'] = data['NPK_sum'] / 3
        
        if 'pH' in data.columns:
            data['pH_category'] = pd.cut(data['pH'], bins=[0, 5.5, 6.5, 7.5, 14],
                                         labels=['Acidic', 'Slightly_Acidic', 'Neutral', 'Alkaline'])
            data['pH_category'] = data['pH_category'].astype(str)
        
        return data
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels if classification
        if self.model_type == 'classification':
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = y_train
        
        # Train model
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            if self.model_type == 'classification':
                y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_val_encoded = y_val
            
            eval_set = [(X_train_scaled, y_train_encoded), (X_val_scaled, y_val_encoded)]
            self.model.fit(X_train_scaled, y_train_encoded, 
                          eval_set=eval_set, verbose=True)
        else:
            self.model.fit(X_train_scaled, y_train_encoded)
        
        self.is_trained = True
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Decode predictions if classification
        if self.model_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)"""
        if self.model_type != 'classification':
            raise ValueError("Probability prediction only available for classification")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return importance_df
        
        return importance
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class FertilizerAdvisor:
    """Complete fertilizer advisory system"""
    
    def __init__(self):
        self.crop_requirements = self._load_crop_requirements()
        self.fertilizer_compositions = self._load_fertilizer_compositions()
        self.organic_fertilizers = self._load_organic_fertilizers()
    
    def _load_crop_requirements(self) -> Dict:
        """Load crop-specific nutrient requirements"""
        return {
            'Rice': {'N': [60, 80], 'P': [30, 40], 'K': [30, 40], 'pH': [5.5, 6.5]},
            'Wheat': {'N': [80, 100], 'P': [40, 50], 'K': [30, 40], 'pH': [6.0, 7.0]},
            'Maize': {'N': [100, 120], 'P': [50, 60], 'K': [40, 50], 'pH': [5.5, 7.0]},
            'Cotton': {'N': [100, 120], 'P': [40, 50], 'K': [40, 50], 'pH': [6.0, 7.5]},
            'Sugarcane': {'N': [200, 250], 'P': [60, 80], 'K': [80, 100], 'pH': [6.0, 7.5]},
            'Potato': {'N': [100, 150], 'P': [60, 80], 'K': [100, 120], 'pH': [5.0, 6.0]},
            'Tomato': {'N': [120, 150], 'P': [50, 70], 'K': [80, 100], 'pH': [6.0, 7.0]},
            'Onion': {'N': [80, 100], 'P': [40, 50], 'K': [60, 80], 'pH': [6.0, 7.0]},
            'Cabbage': {'N': [100, 120], 'P': [50, 60], 'K': [80, 100], 'pH': [6.0, 7.5]},
        }
    
    def _load_fertilizer_compositions(self) -> Dict:
        """Load fertilizer types and compositions"""
        return {
            'Urea': {'N': 46, 'P': 0, 'K': 0},
            'DAP': {'N': 18, 'P': 46, 'K': 0},
            'MOP': {'N': 0, 'P': 0, 'K': 60},
            'NPK_10:26:26': {'N': 10, 'P': 26, 'K': 26},
            'NPK_12:32:16': {'N': 12, 'P': 32, 'K': 16},
            'NPK_20:20:20': {'N': 20, 'P': 20, 'K': 20},
            'NPK_19:19:19': {'N': 19, 'P': 19, 'K': 19},
        }
    
    def _load_organic_fertilizers(self) -> Dict:
        """Load organic fertilizer options"""
        return {
            'FYM': {'N': 0.5, 'P': 0.2, 'K': 0.5, 'rate': '10-15 tons/ha'},
            'Compost': {'N': 0.8, 'P': 0.4, 'K': 0.8, 'rate': '5-10 tons/ha'},
            'Vermicompost': {'N': 1.5, 'P': 1.0, 'K': 1.0, 'rate': '3-5 tons/ha'},
        }
    
    def calculate_nutrient_deficit(self, soil_data: Dict, crop: str) -> Dict:
        """Calculate nutrient deficit for a specific crop"""
        crop_req = self.crop_requirements.get(crop, {})
        
        if not crop_req:
            return {'error': f'Crop {crop} not found in database'}
        
        deficit = {
            'N_deficit': max(0, np.mean(crop_req.get('N', [0])) - soil_data.get('Nitrogen', 0)),
            'P_deficit': max(0, np.mean(crop_req.get('P', [0])) - soil_data.get('Phosphorus', 0)),
            'K_deficit': max(0, np.mean(crop_req.get('K', [0])) - soil_data.get('Potassium', 0)),
            'pH_optimal': crop_req.get('pH', [6.0, 7.0]),
            'pH_current': soil_data.get('pH', 7.0),
        }
        
        # Calculate pH adjustment needed
        if deficit['pH_current'] < deficit['pH_optimal'][0]:
            deficit['pH_adjustment'] = 'Add lime to increase pH'
        elif deficit['pH_current'] > deficit['pH_optimal'][1]:
            deficit['pH_adjustment'] = 'Add sulfur to decrease pH'
        else:
            deficit['pH_adjustment'] = 'pH is optimal'
        
        return deficit
    
    def recommend_fertilizers(self, deficit: Dict, field_area: float = 1.0,
                            prefer_organic: bool = False) -> Dict:
        """Recommend specific fertilizers and quantities"""
        recommendations = []
        
        # Calculate quantities needed
        n_needed = deficit.get('N_deficit', 0)
        p_needed = deficit.get('P_deficit', 0)
        k_needed = deficit.get('K_deficit', 0)
        
        if prefer_organic:
            # Organic recommendations
            recommendations.append({
                'fertilizer': 'Compost',
                'quantity': f"{5 * field_area} tons",
                'nutrients': f"Provides N:{5*0.8*field_area}, P:{5*0.4*field_area}, K:{5*0.8*field_area} kg",
                'application': 'Apply before sowing/planting',
            })
        
        # Nitrogen fertilizer
        if n_needed > 30:
            urea_qty = (n_needed / 46) * 100 * field_area
            recommendations.append({
                'fertilizer': 'Urea',
                'quantity': f"{urea_qty:.1f} kg/ha",
                'nutrients': f"Provides {n_needed * field_area:.1f} kg N",
                'application': 'Split application: 50% basal, 25% at 30 DAP, 25% at 60 DAP',
            })
        
        # Phosphorus fertilizer
        if p_needed > 20:
            dap_qty = (p_needed / 46) * 100 * field_area
            recommendations.append({
                'fertilizer': 'DAP',
                'quantity': f"{dap_qty:.1f} kg/ha",
                'nutrients': f"Provides {p_needed * field_area:.1f} kg P",
                'application': 'Apply as basal dose before sowing',
            })
        
        # Potassium fertilizer
        if k_needed > 20:
            mop_qty = (k_needed / 60) * 100 * field_area
            recommendations.append({
                'fertilizer': 'MOP',
                'quantity': f"{mop_qty:.1f} kg/ha",
                'nutrients': f"Provides {k_needed * field_area:.1f} kg K",
                'application': 'Apply in 2 splits: 50% basal, 50% at flowering',
            })
        
        # Complex fertilizer if all nutrients needed
        if n_needed > 20 and p_needed > 20 and k_needed > 20:
            npk_qty = ((n_needed + p_needed + k_needed) / 60) * 100 * field_area
            recommendations.append({
                'fertilizer': 'NPK 19:19:19',
                'quantity': f"{npk_qty:.1f} kg/ha",
                'nutrients': f"Balanced nutrition",
                'application': 'Apply as basal dose',
            })
        
        return {
            'recommendations': recommendations,
            'deficit_summary': {
                'N_deficit': f"{n_needed:.1f} kg/ha",
                'P_deficit': f"{p_needed:.1f} kg/ha",
                'K_deficit': f"{k_needed:.1f} kg/ha",
            },
            'pH_recommendation': deficit.get('pH_adjustment', 'No adjustment needed'),
        }
    
    def generate_complete_recommendation(self, soil_data: Dict, soil_type: str,
                                        crop: str, field_area: float = 1.0,
                                        growth_stage: str = 'Vegetative',
                                        prefer_organic: bool = False) -> Dict:
        """Generate complete fertilizer recommendation"""
        
        # Calculate deficit
        deficit = self.calculate_nutrient_deficit(soil_data, crop)
        
        # Get fertilizer recommendations
        fertilizers = self.recommend_fertilizers(deficit, field_area, prefer_organic)
        
        # Add soil-specific advice
        soil_advice = self._get_soil_specific_advice(soil_type)
        
        # Add growth stage specific timing
        timing_advice = self._get_growth_stage_timing(crop, growth_stage)
        
        complete_recommendation = {
            'crop': crop,
            'soil_type': soil_type,
            'field_area': f"{field_area} hectare(s)",
            'growth_stage': growth_stage,
            'soil_properties': soil_data,
            'nutrient_deficit': deficit,
            'fertilizer_recommendations': fertilizers['recommendations'],
            'deficit_summary': fertilizers['deficit_summary'],
            'pH_recommendation': fertilizers['pH_recommendation'],
            'soil_specific_advice': soil_advice,
            'timing_advice': timing_advice,
            'additional_tips': self._get_additional_tips(crop, soil_type),
        }
        
        return complete_recommendation
    
    def _get_soil_specific_advice(self, soil_type: str) -> List[str]:
        """Get soil-specific management advice"""
        advice_map = {
            'Sandy': [
                'Apply organic matter to improve water retention',
                'Use frequent light fertilizer applications',
                'Consider drip irrigation for better nutrient use',
            ],
            'Clayey': [
                'Add organic matter to improve drainage',
                'Avoid over-irrigation to prevent waterlogging',
                'Apply fertilizers in split doses',
            ],
            'Loamy': [
                'Ideal soil type, maintain organic matter',
                'Follow standard fertilizer recommendations',
            ],
            'Silty': [
                'Good water retention, monitor drainage',
                'Regular organic matter addition recommended',
            ],
        }
        return advice_map.get(soil_type, ['Follow general best practices'])
    
    def _get_growth_stage_timing(self, crop: str, stage: str) -> str:
        """Get growth stage specific timing advice"""
        timing_map = {
            'Vegetative': 'High nitrogen requirement for leaf and stem development',
            'Flowering': 'Balanced NPK with emphasis on P and K for flower formation',
            'Fruiting': 'High potassium for fruit development and quality',
            'Ripening': 'Minimal fertilization, focus on irrigation management',
        }
        return timing_map.get(stage, 'Apply as per crop requirement')
    
    def _get_additional_tips(self, crop: str, soil_type: str) -> List[str]:
        """Get additional cultivation tips"""
        return [
            'Conduct soil test every season for accurate recommendations',
            'Apply micronutrients based on deficiency symptoms',
            'Maintain proper soil moisture for nutrient uptake',
            'Consider foliar application during critical growth stages',
            'Practice crop rotation for better soil health',
        ]


if __name__ == "__main__":
    # Example usage
    print("Fertilizer Recommendation System initialized\n")
    
    # Create advisor
    advisor = FertilizerAdvisor()
    
    # Example soil data
    soil_data = {
        'pH': 6.5,
        'Nitrogen': 45,
        'Phosphorus': 25,
        'Potassium': 35,
        'Organic_Matter': 2.5,
        'Moisture': 35,
    }
    
    # Generate recommendation
    recommendation = advisor.generate_complete_recommendation(
        soil_data=soil_data,
        soil_type='Loamy',
        crop='Wheat',
        field_area=1.0,
        growth_stage='Vegetative',
        prefer_organic=False
    )
    
    print("Sample Recommendation:")
    print(json.dumps(recommendation, indent=2))
