"""
Multi-Modal Fusion System for Routesit AI
Combines Vision + Text + Accident data + Traffic patterns
Intelligent fusion with uncertainty quantification
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModalityData:
    """Data from a single modality"""
    modality_type: str  # "text", "image", "accident", "traffic"
    features: np.ndarray
    confidence: float
    metadata: Dict[str, Any]
    timestamp: str

@dataclass
class FusedAnalysis:
    """Result of multi-modal fusion"""
    analysis_id: str
    fused_features: np.ndarray
    modality_contributions: Dict[str, float]
    overall_confidence: float
    uncertainty_estimate: float
    recommendations: List[Dict[str, Any]]
    explanation: str
    timestamp: str

class VisionProcessor:
    """Computer vision processing for road images"""
    
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained vision model (simplified)
        self.vision_model = self._load_vision_model()
        
        logger.info("Vision processor initialized")
    
    def _load_vision_model(self):
        """Load pre-trained vision model"""
        try:
            # Simplified vision model (would use actual pre-trained model)
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            
            logger.info("Vision model loaded")
            return model
            
        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            return None
    
    def process_image(self, image_path: str) -> ModalityData:
        """Process road image and extract features"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.vision_model(image_tensor)
                features = features.squeeze().numpy()
            
            # Detect road elements (simplified)
            detected_elements = self._detect_road_elements(image_path)
            
            # Calculate confidence based on detection quality
            confidence = self._calculate_vision_confidence(detected_elements)
            
            return ModalityData(
                modality_type="image",
                features=features,
                confidence=confidence,
                metadata={
                    "detected_elements": detected_elements,
                    "image_size": image.size,
                    "processing_method": "cnn_features"
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return self._create_empty_vision_data()
    
    def _detect_road_elements(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect road elements in image (simplified)"""
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            detected_elements = []
            
            # Detect lines (simplified)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                detected_elements.append({
                    "type": "road_marking",
                    "count": len(lines),
                    "confidence": 0.7
                })
            
            # Detect signs (simplified)
            # This would use actual sign detection in production
            detected_elements.append({
                "type": "traffic_sign",
                "count": 1,  # Placeholder
                "confidence": 0.5
            })
            
            return detected_elements
            
        except Exception as e:
            logger.error(f"Error detecting road elements: {e}")
            return []
    
    def _calculate_vision_confidence(self, detected_elements: List[Dict[str, Any]]) -> float:
        """Calculate confidence based on detection quality"""
        if not detected_elements:
            return 0.3
        
        # Calculate average confidence
        confidences = [elem.get("confidence", 0.5) for elem in detected_elements]
        return np.mean(confidences)
    
    def _create_empty_vision_data(self) -> ModalityData:
        """Create empty vision data when processing fails"""
        return ModalityData(
            modality_type="image",
            features=np.zeros(256),
            confidence=0.1,
            metadata={"error": "Image processing failed"},
            timestamp=datetime.now().isoformat()
        )

class TextProcessor:
    """Text processing for road safety descriptions"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.domain_keywords = self._load_domain_keywords()
        
        logger.info("Text processor initialized")
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load road safety domain keywords"""
        return {
            "interventions": [
                "zebra crossing", "speed limit", "traffic light", "speed hump",
                "warning sign", "barrier", "pedestrian bridge", "advance warning"
            ],
            "problems": [
                "faded", "damaged", "missing", "broken", "worn", "unclear",
                "dangerous", "unsafe", "hazardous"
            ],
            "locations": [
                "school zone", "hospital", "intersection", "highway", "urban",
                "rural", "bridge", "tunnel"
            ],
            "severity": [
                "fatal", "injury", "accident", "collision", "crash", "incident"
            ]
        }
    
    def process_text(self, text: str) -> ModalityData:
        """Process text description and extract features"""
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(text)
            
            # Extract domain-specific features
            domain_features = self._extract_domain_features(text)
            
            # Combine embeddings and domain features
            combined_features = np.concatenate([embeddings, domain_features])
            
            # Calculate confidence based on text quality
            confidence = self._calculate_text_confidence(text, domain_features)
            
            return ModalityData(
                modality_type="text",
                features=combined_features,
                confidence=confidence,
                metadata={
                    "text_length": len(text),
                    "domain_features": domain_features.tolist(),
                    "processing_method": "sentence_transformer"
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return self._create_empty_text_data()
    
    def _extract_domain_features(self, text: str) -> np.ndarray:
        """Extract domain-specific features from text"""
        features = []
        text_lower = text.lower()
        
        # Count keyword occurrences
        for category, keywords in self.domain_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features.append(count)
        
        # Text quality features
        features.append(len(text.split()))  # Word count
        features.append(len(text))  # Character count
        features.append(text.count('!'))  # Exclamation marks (urgency)
        features.append(text.count('?'))  # Question marks (uncertainty)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_text_confidence(self, text: str, domain_features: np.ndarray) -> float:
        """Calculate confidence based on text quality"""
        # Base confidence
        confidence = 0.5
        
        # Increase confidence for domain-relevant text
        if np.sum(domain_features[:4]) > 0:  # Has domain keywords
            confidence += 0.3
        
        # Increase confidence for longer, more descriptive text
        if len(text.split()) > 10:
            confidence += 0.1
        
        # Decrease confidence for very short text
        if len(text.split()) < 3:
            confidence -= 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _create_empty_text_data(self) -> ModalityData:
        """Create empty text data when processing fails"""
        return ModalityData(
            modality_type="text",
            features=np.zeros(384 + 8),  # 384 for embeddings + 8 for domain features
            confidence=0.1,
            metadata={"error": "Text processing failed"},
            timestamp=datetime.now().isoformat()
        )

class AccidentDataProcessor:
    """Process accident data for fusion"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'severity_fatal', 'severity_injury', 'severity_property',
            'road_type_highway', 'road_type_urban', 'road_type_rural',
            'weather_clear', 'weather_rain', 'weather_fog',
            'traffic_volume_high', 'traffic_volume_medium', 'traffic_volume_low',
            'intervention_count', 'accident_frequency'
        ]
        
        logger.info("Accident data processor initialized")
    
    def process_accident_data(self, accident_data: Dict[str, Any]) -> ModalityData:
        """Process accident data and extract features"""
        try:
            # Extract features from accident data
            features = self._extract_accident_features(accident_data)
            
            # Normalize features
            features_normalized = self.scaler.fit_transform([features])[0]
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_accident_confidence(accident_data)
            
            return ModalityData(
                modality_type="accident",
                features=features_normalized,
                confidence=confidence,
                metadata={
                    "data_completeness": self._calculate_completeness(accident_data),
                    "data_source": accident_data.get("source", "unknown"),
                    "processing_method": "statistical_features"
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing accident data: {e}")
            return self._create_empty_accident_data()
    
    def _extract_accident_features(self, accident_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from accident data"""
        features = []
        
        # Severity features (one-hot encoding)
        severity = accident_data.get("severity", "unknown")
        features.extend([
            1.0 if severity == "fatal" else 0.0,
            1.0 if severity == "injury" else 0.0,
            1.0 if severity == "property" else 0.0
        ])
        
        # Road type features
        road_type = accident_data.get("road_type", "unknown")
        features.extend([
            1.0 if road_type == "highway" else 0.0,
            1.0 if road_type == "urban" else 0.0,
            1.0 if road_type == "rural" else 0.0
        ])
        
        # Weather features
        weather = accident_data.get("weather", "unknown")
        features.extend([
            1.0 if weather == "clear" else 0.0,
            1.0 if weather == "rain" else 0.0,
            1.0 if weather == "fog" else 0.0
        ])
        
        # Traffic volume features
        traffic_volume = accident_data.get("traffic_volume", "unknown")
        features.extend([
            1.0 if traffic_volume == "high" else 0.0,
            1.0 if traffic_volume == "medium" else 0.0,
            1.0 if traffic_volume == "low" else 0.0
        ])
        
        # Intervention count
        interventions = accident_data.get("interventions_present", [])
        features.append(float(len(interventions)))
        
        # Accident frequency (simplified)
        features.append(1.0)  # Placeholder for accident frequency
        
        return features
    
    def _calculate_accident_confidence(self, accident_data: Dict[str, Any]) -> float:
        """Calculate confidence based on data quality"""
        confidence = 0.5
        
        # Increase confidence for verified data
        if accident_data.get("verified", False):
            confidence += 0.3
        
        # Increase confidence for complete data
        completeness = self._calculate_completeness(accident_data)
        confidence += completeness * 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _calculate_completeness(self, accident_data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        required_fields = ["severity", "road_type", "weather", "traffic_volume"]
        present_fields = sum(1 for field in required_fields if field in accident_data)
        return present_fields / len(required_fields)
    
    def _create_empty_accident_data(self) -> ModalityData:
        """Create empty accident data when processing fails"""
        return ModalityData(
            modality_type="accident",
            features=np.zeros(len(self.feature_columns)),
            confidence=0.1,
            metadata={"error": "Accident data processing failed"},
            timestamp=datetime.now().isoformat()
        )

class TrafficPatternProcessor:
    """Process traffic pattern data"""
    
    def __init__(self):
        self.pattern_features = [
            'peak_hour_traffic', 'weekend_traffic', 'seasonal_variation',
            'vehicle_mix', 'speed_distribution', 'congestion_level'
        ]
        
        logger.info("Traffic pattern processor initialized")
    
    def process_traffic_patterns(self, traffic_data: Dict[str, Any]) -> ModalityData:
        """Process traffic pattern data"""
        try:
            # Extract traffic features
            features = self._extract_traffic_features(traffic_data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_traffic_confidence(traffic_data)
            
            return ModalityData(
                modality_type="traffic",
                features=features,
                confidence=confidence,
                metadata={
                    "data_period": traffic_data.get("period", "unknown"),
                    "data_source": traffic_data.get("source", "unknown"),
                    "processing_method": "pattern_analysis"
                },
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing traffic patterns: {e}")
            return self._create_empty_traffic_data()
    
    def _extract_traffic_features(self, traffic_data: Dict[str, Any]) -> np.ndarray:
        """Extract traffic pattern features"""
        features = []
        
        # Peak hour traffic
        features.append(traffic_data.get("peak_hour_traffic", 0.5))
        
        # Weekend traffic
        features.append(traffic_data.get("weekend_traffic", 0.5))
        
        # Seasonal variation
        features.append(traffic_data.get("seasonal_variation", 0.5))
        
        # Vehicle mix
        features.append(traffic_data.get("vehicle_mix", 0.5))
        
        # Speed distribution
        features.append(traffic_data.get("speed_distribution", 0.5))
        
        # Congestion level
        features.append(traffic_data.get("congestion_level", 0.5))
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_traffic_confidence(self, traffic_data: Dict[str, Any]) -> float:
        """Calculate confidence based on traffic data quality"""
        confidence = 0.5
        
        # Increase confidence for recent data
        if "timestamp" in traffic_data:
            confidence += 0.2
        
        # Increase confidence for comprehensive data
        if len(traffic_data) > 5:
            confidence += 0.2
        
        return min(max(confidence, 0.1), 1.0)
    
    def _create_empty_traffic_data(self) -> ModalityData:
        """Create empty traffic data when processing fails"""
        return ModalityData(
            modality_type="traffic",
            features=np.zeros(len(self.pattern_features)),
            confidence=0.1,
            metadata={"error": "Traffic pattern processing failed"},
            timestamp=datetime.now().isoformat()
        )

class MultiModalFusion:
    """
    Multi-modal fusion system combining all modalities
    """
    
    def __init__(self):
        self.vision_processor = VisionProcessor()
        self.text_processor = TextProcessor()
        self.accident_processor = AccidentDataProcessor()
        self.traffic_processor = TrafficPatternProcessor()
        
        # Fusion model
        self.fusion_model = self._create_fusion_model()
        
        # Data storage paths
        self.models_path = Path("models/multimodal_fusion")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Multi-modal fusion system initialized")
    
    def _create_fusion_model(self) -> nn.Module:
        """Create neural network for multi-modal fusion"""
        class FusionNetwork(nn.Module):
            def __init__(self):
                super(FusionNetwork, self).__init__()
                
                # Modality-specific encoders
                self.vision_encoder = nn.Linear(256, 128)
                self.text_encoder = nn.Linear(392, 128)  # 384 + 8
                self.accident_encoder = nn.Linear(14, 128)
                self.traffic_encoder = nn.Linear(6, 128)
                
                # Cross-modal attention
                self.attention = nn.MultiheadAttention(128, 8, batch_first=True)
                
                # Fusion layers
                self.fusion_layer = nn.Linear(128 * 4, 256)
                self.output_layer = nn.Linear(256, 128)
                
                # Dropout for regularization
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, vision_feat, text_feat, accident_feat, traffic_feat):
                # Encode each modality
                vision_encoded = F.relu(self.vision_encoder(vision_feat))
                text_encoded = F.relu(self.text_encoder(text_feat))
                accident_encoded = F.relu(self.accident_encoder(accident_feat))
                traffic_encoded = F.relu(self.traffic_encoder(traffic_feat))
                
                # Stack for attention
                modalities = torch.stack([vision_encoded, text_encoded, accident_encoded, traffic_encoded], dim=1)
                
                # Cross-modal attention
                attended, _ = self.attention(modalities, modalities, modalities)
                
                # Flatten and fuse
                fused = torch.flatten(attended, start_dim=1)
                fused = F.relu(self.fusion_layer(fused))
                fused = self.dropout(fused)
                output = self.output_layer(fused)
                
                return output
        
        return FusionNetwork()
    
    def fuse_modalities(self, 
                       text: str = None,
                       image_path: str = None,
                       accident_data: Dict[str, Any] = None,
                       traffic_data: Dict[str, Any] = None) -> FusedAnalysis:
        """
        Fuse multiple modalities into unified analysis
        """
        try:
            modality_data = []
            
            # Process text if provided
            if text:
                text_data = self.text_processor.process_text(text)
                modality_data.append(text_data)
            
            # Process image if provided
            if image_path and os.path.exists(image_path):
                vision_data = self.vision_processor.process_image(image_path)
                modality_data.append(vision_data)
            
            # Process accident data if provided
            if accident_data:
                accident_processed = self.accident_processor.process_accident_data(accident_data)
                modality_data.append(accident_processed)
            
            # Process traffic data if provided
            if traffic_data:
                traffic_processed = self.traffic_processor.process_traffic_patterns(traffic_data)
                modality_data.append(traffic_processed)
            
            if not modality_data:
                return self._create_empty_fusion_result()
            
            # Fuse modalities
            fused_features, modality_contributions = self._fuse_features(modality_data)
            
            # Calculate overall confidence and uncertainty
            overall_confidence = self._calculate_overall_confidence(modality_data)
            uncertainty = self._calculate_uncertainty(modality_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(fused_features, modality_data)
            
            # Create explanation
            explanation = self._create_explanation(modality_data, modality_contributions)
            
            return FusedAnalysis(
                analysis_id=f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                fused_features=fused_features,
                modality_contributions=modality_contributions,
                overall_confidence=overall_confidence,
                uncertainty_estimate=uncertainty,
                recommendations=recommendations,
                explanation=explanation,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in multi-modal fusion: {e}")
            return self._create_empty_fusion_result()
    
    def _fuse_features(self, modality_data: List[ModalityData]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Fuse features from different modalities"""
        try:
            # Prepare features for fusion model
            vision_feat = torch.zeros(256)
            text_feat = torch.zeros(392)
            accident_feat = torch.zeros(14)
            traffic_feat = torch.zeros(6)
            
            modality_contributions = {}
            
            # Assign features based on modality type
            for data in modality_data:
                if data.modality_type == "image":
                    vision_feat = torch.tensor(data.features, dtype=torch.float32)
                    modality_contributions["vision"] = data.confidence
                elif data.modality_type == "text":
                    text_feat = torch.tensor(data.features, dtype=torch.float32)
                    modality_contributions["text"] = data.confidence
                elif data.modality_type == "accident":
                    accident_feat = torch.tensor(data.features, dtype=torch.float32)
                    modality_contributions["accident"] = data.confidence
                elif data.modality_type == "traffic":
                    traffic_feat = torch.tensor(data.features, dtype=torch.float32)
                    modality_contributions["traffic"] = data.confidence
            
            # Fuse using neural network
            with torch.no_grad():
                fused_features = self.fusion_model(
                    vision_feat.unsqueeze(0),
                    text_feat.unsqueeze(0),
                    accident_feat.unsqueeze(0),
                    traffic_feat.unsqueeze(0)
                )
                fused_features = fused_features.squeeze().numpy()
            
            return fused_features, modality_contributions
            
        except Exception as e:
            logger.error(f"Error fusing features: {e}")
            return np.zeros(128), {}
    
    def _calculate_overall_confidence(self, modality_data: List[ModalityData]) -> float:
        """Calculate overall confidence from all modalities"""
        if not modality_data:
            return 0.0
        
        confidences = [data.confidence for data in modality_data]
        
        # Weighted average based on modality importance
        weights = {
            "text": 0.3,
            "image": 0.3,
            "accident": 0.25,
            "traffic": 0.15
        }
        
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for data in modality_data:
            weight = weights.get(data.modality_type, 0.1)
            weighted_confidence += data.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_uncertainty(self, modality_data: List[ModalityData]) -> float:
        """Calculate uncertainty estimate"""
        if len(modality_data) < 2:
            return 0.5  # High uncertainty with single modality
        
        # Calculate variance in confidences
        confidences = [data.confidence for data in modality_data]
        confidence_variance = np.var(confidences)
        
        # Calculate uncertainty based on variance and number of modalities
        uncertainty = confidence_variance + (1.0 / len(modality_data))
        
        return min(max(uncertainty, 0.0), 1.0)
    
    def _generate_recommendations(self, fused_features: np.ndarray, modality_data: List[ModalityData]) -> List[Dict[str, Any]]:
        """Generate recommendations based on fused analysis"""
        recommendations = []
        
        # Analyze fused features to generate recommendations
        feature_magnitude = np.linalg.norm(fused_features)
        
        if feature_magnitude > 0.7:
            recommendations.append({
                "type": "high_priority",
                "description": "High-priority intervention recommended",
                "confidence": 0.8,
                "reasoning": "Strong signal from multiple modalities"
            })
        elif feature_magnitude > 0.4:
            recommendations.append({
                "type": "medium_priority",
                "description": "Medium-priority intervention recommended",
                "confidence": 0.6,
                "reasoning": "Moderate signal from available modalities"
            })
        else:
            recommendations.append({
                "type": "low_priority",
                "description": "Low-priority intervention recommended",
                "confidence": 0.4,
                "reasoning": "Weak signal from available modalities"
            })
        
        # Add modality-specific recommendations
        for data in modality_data:
            if data.modality_type == "image" and data.confidence > 0.7:
                recommendations.append({
                    "type": "visual_analysis",
                    "description": "Visual analysis supports intervention need",
                    "confidence": data.confidence,
                    "reasoning": "High-confidence visual detection"
                })
            
            elif data.modality_type == "accident" and data.confidence > 0.7:
                recommendations.append({
                    "type": "accident_history",
                    "description": "Accident history supports intervention need",
                    "confidence": data.confidence,
                    "reasoning": "High-confidence accident data"
                })
        
        return recommendations
    
    def _create_explanation(self, modality_data: List[ModalityData], modality_contributions: Dict[str, float]) -> str:
        """Create human-readable explanation of fusion process"""
        explanation_parts = []
        
        explanation_parts.append("Multi-modal analysis combining:")
        
        for data in modality_data:
            modality_name = data.modality_type.title()
            confidence = data.confidence
            explanation_parts.append(f"- {modality_name} analysis (confidence: {confidence:.2f})")
        
        if modality_contributions:
            explanation_parts.append("\nModality contributions:")
            for modality, contribution in modality_contributions.items():
                explanation_parts.append(f"- {modality.title()}: {contribution:.2f}")
        
        return "\n".join(explanation_parts)
    
    def _create_empty_fusion_result(self) -> FusedAnalysis:
        """Create empty fusion result when no modalities available"""
        return FusedAnalysis(
            analysis_id=f"empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            fused_features=np.zeros(128),
            modality_contributions={},
            overall_confidence=0.0,
            uncertainty_estimate=1.0,
            recommendations=[{
                "type": "insufficient_data",
                "description": "Insufficient data for analysis",
                "confidence": 0.0,
                "reasoning": "No modalities provided"
            }],
            explanation="No modalities provided for fusion",
            timestamp=datetime.now().isoformat()
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "vision_model_available": self.vision_processor.vision_model is not None,
            "text_embedding_dim": 392,
            "accident_feature_dim": 14,
            "traffic_feature_dim": 6,
            "fusion_output_dim": 128,
            "supported_modalities": ["text", "image", "accident", "traffic"]
        }

# Global instance
multimodal_fusion = None

def get_multimodal_fusion() -> MultiModalFusion:
    """Get global multimodal fusion instance"""
    global multimodal_fusion
    if multimodal_fusion is None:
        multimodal_fusion = MultiModalFusion()
    return multimodal_fusion

def fuse_multimodal_data(text: str = None,
                         image_path: str = None,
                         accident_data: Dict[str, Any] = None,
                         traffic_data: Dict[str, Any] = None) -> FusedAnalysis:
    """Convenience function for multi-modal fusion"""
    fusion_system = get_multimodal_fusion()
    return fusion_system.fuse_modalities(text, image_path, accident_data, traffic_data)
