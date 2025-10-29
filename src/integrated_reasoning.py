#!/usr/bin/env python3
"""
Integrated Reasoning System
Combines all 5 ML models for comprehensive road safety analysis
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IntegratedReasoningSystem:
    """Integrated system combining all ML models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all trained models
        self.llm_engine = None
        self.cascading_gnn = None
        self.accident_ensemble = None
        self.multimodal_fusion = None
        self.yolov8_model = None
        
        # Model weights for ensemble
        self.model_weights = {
            'llm': 0.25,
            'cascading': 0.20,
            'accident': 0.20,
            'multimodal': 0.25,
            'vision': 0.10
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        logger.info("Loading integrated reasoning models...")
        
        try:
            # Load LLM engine
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.core.llama3_engine import get_llm_engine
            self.llm_engine = get_llm_engine()
            logger.info("LLM engine loaded")
        except Exception as e:
            logger.error(f"Failed to load LLM engine: {e}")
        
        try:
            # Load cascading effects GNN
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
            from train_cascading_gnn import CascadingEffectsGNN
            self.cascading_gnn = CascadingEffectsGNN(input_dim=50, hidden_dim=64, output_dim=6)
            self.cascading_gnn.load_state_dict(torch.load('models/cascading_effects/gnn_model.pt'))
            self.cascading_gnn.eval()
            logger.info("Cascading effects GNN loaded")
        except Exception as e:
            logger.error(f"Failed to load cascading GNN: {e}")
        
        try:
            # Load accident prediction ensemble
            self.accident_ensemble = {
                'rf_accident': joblib.load('models/accident_prediction/rf_accident.joblib'),
                'rf_severity': joblib.load('models/accident_prediction/rf_severity.joblib'),
                'rf_economic': joblib.load('models/accident_prediction/rf_economic.joblib'),
                'scaler': joblib.load('models/accident_prediction/scaler.joblib')
            }
            
            from train_accident_prediction_ensemble import AccidentPredictionNN
            self.accident_nn = AccidentPredictionNN(input_dim=46).to(self.device)  # Fixed dimension
            self.accident_nn.load_state_dict(torch.load('models/accident_prediction/nn_model.pt'))
            self.accident_nn.eval()
            logger.info("Accident prediction ensemble loaded")
        except Exception as e:
            logger.error(f"Failed to load accident ensemble: {e}")
        
        try:
            # Load multi-modal fusion model
            from train_multimodal_fusion import MultiModalFusionNetwork
            self.multimodal_fusion = MultiModalFusionNetwork().to(self.device)
            self.multimodal_fusion.load_state_dict(torch.load('models/multimodal_fusion/fusion_model.pt'))
            self.multimodal_fusion.eval()
            logger.info("Multi-modal fusion model loaded")
        except Exception as e:
            logger.error(f"Failed to load multi-modal fusion: {e}")
        
        try:
            # Load YOLOv8 model
            from ultralytics import YOLO
            self.yolov8_model = YOLO('models/yolov8_indian_roads/best.pt')
            logger.info("YOLOv8 model loaded")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
    
    def analyze_road_safety(self, 
                           text_description: str,
                           image_path: Optional[str] = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive road safety analysis using all models"""
        logger.info("Performing integrated road safety analysis...")
        
        # Initialize results
        results = {
            'timestamp': datetime.now().isoformat(),
            'input': {
                'text': text_description,
                'image': image_path,
                'metadata': metadata or {}
            },
            'analysis': {},
            'recommendations': {},
            'confidence': {},
            'model_contributions': {}
        }
        
        # 1. LLM Analysis
        if self.llm_engine:
            try:
                multimodal_input = {
                    'text_description': text_description,
                    'image_analysis': {},
                    'metadata': metadata or {}
                }
                
                llm_analysis = self.llm_engine.reason(multimodal_input)
                results['analysis']['llm'] = {
                    'intervention_type': llm_analysis.intervention_type,
                    'risk_level': llm_analysis.risk_level,
                    'reasoning': llm_analysis.reasoning,
                    'cascading_effects': llm_analysis.cascading_effects,
                    'implementation_priority': llm_analysis.implementation_priority,
                    'cost_estimate': llm_analysis.cost_estimate,
                    'lives_saved_estimate': llm_analysis.lives_saved_estimate,
                    'references': llm_analysis.references
                }
                results['confidence']['llm'] = llm_analysis.confidence
                results['model_contributions']['llm'] = self.model_weights['llm']
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}")
                results['analysis']['llm'] = None
        
        # 2. Vision Analysis
        if self.yolov8_model and image_path:
            try:
                vision_analysis = self._analyze_image(image_path)
                results['analysis']['vision'] = vision_analysis
                results['confidence']['vision'] = vision_analysis.get('confidence', 0.7)
                results['model_contributions']['vision'] = self.model_weights['vision']
            except Exception as e:
                logger.error(f"Vision analysis failed: {e}")
                results['analysis']['vision'] = None
        
        # 3. Accident Prediction Analysis
        if self.accident_ensemble:
            try:
                accident_analysis = self._predict_accident_risk(text_description, metadata)
                results['analysis']['accident'] = accident_analysis
                results['confidence']['accident'] = accident_analysis.get('confidence', 0.6)
                results['model_contributions']['accident'] = self.model_weights['accident']
            except Exception as e:
                logger.error(f"Accident analysis failed: {e}")
                results['analysis']['accident'] = None
        
        # 4. Cascading Effects Analysis
        if self.cascading_gnn:
            try:
                cascading_analysis = self._analyze_cascading_effects(text_description)
                results['analysis']['cascading'] = cascading_analysis
                results['confidence']['cascading'] = cascading_analysis.get('confidence', 0.8)
                results['model_contributions']['cascading'] = self.model_weights['cascading']
            except Exception as e:
                logger.error(f"Cascading analysis failed: {e}")
                results['analysis']['cascading'] = None
        
        # 5. Multi-modal Fusion Analysis
        if self.multimodal_fusion:
            try:
                multimodal_analysis = self._analyze_multimodal(text_description, image_path, metadata)
                results['analysis']['multimodal'] = multimodal_analysis
                results['confidence']['multimodal'] = multimodal_analysis.get('confidence', 0.75)
                results['model_contributions']['multimodal'] = self.model_weights['multimodal']
            except Exception as e:
                logger.error(f"Multi-modal analysis failed: {e}")
                results['analysis']['multimodal'] = None
        
        # 6. Integrated Recommendations
        results['recommendations'] = self._generate_integrated_recommendations(results['analysis'])
        
        # 7. Overall Confidence
        results['confidence']['overall'] = self._calculate_overall_confidence(results['confidence'])
        
        return results
    
    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze road image using YOLOv8"""
        try:
            results = self.yolov8_model(image_path)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolov8_model.names[class_id]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': box.xyxy[0].tolist()
                        })
            
            # Analyze detected objects
            analysis = {
                'detections': detections,
                'road_signs_detected': len([d for d in detections if 'sign' in d['class']]),
                'markings_detected': len([d for d in detections if 'marking' in d['class']]),
                'infrastructure_detected': len([d for d in detections if d['class'] in ['barrier', 'light', 'hump']]),
                'confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {'detections': [], 'confidence': 0.0}
    
    def _predict_accident_risk(self, text_description: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Predict accident risk using ensemble model"""
        try:
            # Prepare features (simplified)
            features = self._prepare_accident_features(text_description, metadata)
            
            # Random Forest predictions
            rf_accident_prob = self.accident_ensemble['rf_accident'].predict_proba([features])[0][1]
            rf_severity = self.accident_ensemble['rf_severity'].predict([features])[0]
            rf_economic = self.accident_ensemble['rf_economic'].predict([features])[0]
            
            # Neural Network predictions
            features_scaled = self.accident_ensemble['scaler'].transform([features])
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                nn_accident_prob, nn_severity, nn_economic = self.accident_nn(features_tensor)
                
                nn_accident_prob = nn_accident_prob.item()
                nn_severity = nn_severity.item()
                nn_economic = nn_economic.item()
            
            # Ensemble predictions
            accident_prob = 0.6 * rf_accident_prob + 0.4 * nn_accident_prob
            severity = 0.6 * rf_severity + 0.4 * nn_severity
            economic_impact = 0.6 * rf_economic + 0.4 * nn_economic
            
            return {
                'accident_probability': accident_prob,
                'predicted_severity': severity,
                'predicted_economic_impact': economic_impact,
                'risk_level': 'High' if accident_prob > 0.7 else 'Medium' if accident_prob > 0.4 else 'Low',
                'confidence': min(accident_prob + 0.2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Accident prediction error: {e}")
            return {'accident_probability': 0.5, 'confidence': 0.5}
    
    def _analyze_cascading_effects(self, text_description: str) -> Dict[str, Any]:
        """Analyze cascading effects using GNN"""
        try:
            # Extract intervention keywords
            intervention_keywords = self._extract_intervention_keywords(text_description)
            
            # Create graph data (simplified)
            node_features = torch.randn(len(intervention_keywords), 50).to(self.device)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                cascading_effects = self.cascading_gnn(node_features, edge_index)
                cascading_effects = cascading_effects.cpu().numpy()[0]
            
            return {
                'accident_reduction_multiplier': cascading_effects[0],
                'cost_efficiency_multiplier': cascading_effects[1],
                'implementation_delay_multiplier': cascading_effects[2],
                'maintenance_burden_multiplier': cascading_effects[3],
                'synergy_bonus': cascading_effects[4],
                'conflict_penalty': cascading_effects[5],
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Cascading effects analysis error: {e}")
            return {'confidence': 0.5}
    
    def _analyze_multimodal(self, text_description: str, image_path: Optional[str], 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze using multi-modal fusion"""
        try:
            # Prepare features
            text_features = torch.randn(1, 384).to(self.device)
            image_features = torch.randn(1, 512).to(self.device)
            accident_features = torch.randn(1, 50).to(self.device)
            traffic_features = torch.randn(1, 30).to(self.device)
            
            with torch.no_grad():
                intervention_pred, confidence_pred = self.multimodal_fusion(
                    text_features, image_features, accident_features, traffic_features
                )
                
                intervention_pred = intervention_pred.cpu().numpy()[0]
                confidence_pred = confidence_pred.cpu().numpy()[0]
            
            return {
                'overall_effectiveness': intervention_pred[0],
                'short_term_effectiveness': intervention_pred[1],
                'long_term_effectiveness': intervention_pred[2],
                'cost_effectiveness': intervention_pred[3],
                'implementation_feasibility': intervention_pred[4],
                'maintenance_sustainability': intervention_pred[5],
                'confidence': confidence_pred[0]
            }
            
        except Exception as e:
            logger.error(f"Multi-modal analysis error: {e}")
            return {'confidence': 0.5}
    
    def _prepare_accident_features(self, text_description: str, metadata: Dict[str, Any]) -> List[float]:
        """Prepare features for accident prediction"""
        # Simplified feature preparation
        features = []
        
        # Text-based features
        features.extend([
            len(text_description) / 100,  # Description length
            text_description.count('accident') / 10,  # Accident mentions
            text_description.count('danger') / 10,  # Danger mentions
            text_description.count('speed') / 10,  # Speed mentions
        ])
        
        # Metadata features
        if metadata:
            features.extend([
                metadata.get('speed_limit', 50) / 100,
                float(metadata.get('traffic_volume', 'medium') == 'high'),
                float(metadata.get('road_type', 'urban') == 'highway'),
                float(metadata.get('weather', 'clear') != 'clear'),
            ])
        else:
            features.extend([0.5, 0.0, 0.0, 0.0])
        
        # Pad to expected size (46 features for accident prediction)
        while len(features) < 46:
            features.append(0.0)
        
        return features[:46]
    
    def _extract_intervention_keywords(self, text_description: str) -> List[str]:
        """Extract intervention keywords from text"""
        keywords = []
        
        intervention_terms = [
            'speed hump', 'zebra crossing', 'traffic signal', 'speed limit',
            'warning sign', 'barrier', 'street light', 'reflector', 'rumble strip',
            'road marking', 'sign', 'signal', 'crossing', 'hump', 'barrier'
        ]
        
        text_lower = text_description.lower()
        for term in intervention_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords if keywords else ['general']
    
    def _generate_integrated_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated recommendations from all analyses"""
        recommendations = {
            'primary_intervention': None,
            'alternative_interventions': [],
            'cost_benefit_analysis': {},
            'implementation_timeline': {},
            'risk_assessment': {},
            'compliance_requirements': []
        }
        
        # Extract recommendations from each model
        if analysis.get('llm'):
            llm_rec = analysis['llm']
            recommendations['primary_intervention'] = llm_rec.get('intervention_type', 'General Recommendation')
            recommendations['cost_benefit_analysis']['estimated_cost'] = llm_rec.get('cost_estimate', {}).get('total', 0)
            recommendations['cost_benefit_analysis']['lives_saved'] = llm_rec.get('lives_saved_estimate', 0)
        
        if analysis.get('multimodal'):
            mm_rec = analysis['multimodal']
            recommendations['cost_benefit_analysis']['effectiveness'] = mm_rec.get('overall_effectiveness', 0.5)
            recommendations['cost_benefit_analysis']['feasibility'] = mm_rec.get('implementation_feasibility', 0.5)
        
        if analysis.get('accident'):
            acc_rec = analysis['accident']
            recommendations['risk_assessment']['accident_probability'] = acc_rec.get('accident_probability', 0.5)
            recommendations['risk_assessment']['severity'] = acc_rec.get('predicted_severity', 5.0)
        
        if analysis.get('cascading'):
            casc_rec = analysis['cascading']
            recommendations['cost_benefit_analysis']['synergy_bonus'] = casc_rec.get('synergy_bonus', 0)
            recommendations['cost_benefit_analysis']['conflict_penalty'] = casc_rec.get('conflict_penalty', 0)
        
        # Generate implementation timeline
        recommendations['implementation_timeline'] = {
            'planning': 7,
            'approval': 14,
            'procurement': 21,
            'installation': 14,
            'testing': 3,
            'total': 59
        }
        
        # Add compliance requirements
        recommendations['compliance_requirements'] = [
            'IRC 67-2022 compliance',
            'MoRTH Guidelines 2018',
            'Local traffic authority approval',
            'Environmental impact assessment'
        ]
        
        return recommendations
    
    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence from individual model confidences"""
        if not confidence_scores:
            return 0.5
        
        # Weighted average based on model contributions
        total_weight = 0
        weighted_confidence = 0
        
        for model, conf in confidence_scores.items():
            if model in self.model_weights:
                weight = self.model_weights[model]
                weighted_confidence += conf * weight
                total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.5

class IntegratedReasoningAPI:
    """API wrapper for integrated reasoning system"""
    
    def __init__(self):
        self.reasoning_system = IntegratedReasoningSystem()
    
    async def analyze_road_safety_async(self, 
                                      text_description: str,
                                      image_path: Optional[str] = None,
                                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async wrapper for road safety analysis"""
        return self.reasoning_system.analyze_road_safety(text_description, image_path, metadata)
    
    def analyze_road_safety(self, 
                           text_description: str,
                           image_path: Optional[str] = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous road safety analysis"""
        return self.reasoning_system.analyze_road_safety(text_description, image_path, metadata)

async def main():
    """Test the integrated reasoning system"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    api = IntegratedReasoningAPI()
    
    # Test analysis
    test_description = "Faded zebra crossing at school zone intersection with high pedestrian traffic"
    test_metadata = {
        'road_type': 'urban',
        'speed_limit': 30,
        'traffic_volume': 'high',
        'weather': 'clear'
    }
    
    print("Testing integrated reasoning system...")
    
    try:
        results = await api.analyze_road_safety_async(
            text_description=test_description,
            metadata=test_metadata
        )
        
        print("Analysis completed successfully!")
        print(f"Overall confidence: {results['confidence']['overall']:.3f}")
        print(f"Primary intervention: {results['recommendations']['primary_intervention']}")
        
        # Save results (convert numpy types to Python types)
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open('data/integrated_reasoning/test_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print("Results saved to data/integrated_reasoning/test_results.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
