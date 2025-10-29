"""
Hybrid Cascading Effects System for Routesit AI
Combines rule-based dependencies with Graph Neural Network predictions
Predicts secondary effects and intervention interactions
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CascadingEffect:
    """Cascading effect prediction"""
    effect_id: str
    primary_intervention: str
    secondary_effects: List[Dict[str, Any]]
    dependencies: List[str]
    conflicts: List[str]
    synergies: List[str]
    confidence: float
    prediction_method: str  # "rule_based", "ml_based", "hybrid"
    explanation: str

@dataclass
class RoadNetworkNode:
    """Node in road network graph"""
    node_id: str
    node_type: str  # "intersection", "road_segment", "intervention"
    features: Dict[str, Any]
    location: Dict[str, float]
    interventions: List[str]

@dataclass
class RoadNetworkEdge:
    """Edge in road network graph"""
    source_id: str
    target_id: str
    edge_type: str  # "connects", "influences", "conflicts"
    weight: float
    features: Dict[str, Any]

class RuleBasedDependencies:
    """Rule-based system for intervention dependencies"""
    
    def __init__(self):
        self.dependency_rules = self._load_dependency_rules()
        self.conflict_rules = self._load_conflict_rules()
        self.synergy_rules = self._load_synergy_rules()
        
        logger.info(f"Loaded {len(self.dependency_rules)} dependency rules")
        logger.info(f"Loaded {len(self.conflict_rules)} conflict rules")
        logger.info(f"Loaded {len(self.synergy_rules)} synergy rules")
    
    def _load_dependency_rules(self) -> Dict[str, List[str]]:
        """Load IRC/MoRTH compliance dependency rules"""
        return {
            "zebra_crossing": [
                "advance_warning_sign",
                "speed_limit_sign",
                "street_lighting"
            ],
            "speed_hump": [
                "advance_warning_sign",
                "speed_limit_sign",
                "rumble_strip"
            ],
            "traffic_light": [
                "power_supply",
                "traffic_sensor",
                "backup_power"
            ],
            "school_zone_sign": [
                "speed_limit_sign",
                "advance_warning_sign",
                "speed_hump"
            ],
            "hospital_zone_sign": [
                "speed_limit_sign",
                "advance_warning_sign",
                "priority_lane"
            ],
            "speed_camera": [
                "speed_limit_sign",
                "advance_warning_sign",
                "data_connectivity"
            ],
            "pedestrian_bridge": [
                "approach_ramps",
                "handrails",
                "lighting"
            ],
            "barrier": [
                "foundation",
                "drainage",
                "maintenance_access"
            ]
        }
    
    def _load_conflict_rules(self) -> Dict[str, List[str]]:
        """Load intervention conflict rules"""
        return {
            "speed_hump": [
                "ambulance_route",
                "fire_truck_route",
                "bus_route"
            ],
            "barrier": [
                "emergency_access",
                "maintenance_vehicle_access"
            ],
            "traffic_light": [
                "pedestrian_priority_signal"
            ],
            "speed_camera": [
                "privacy_zone"
            ]
        }
    
    def _load_synergy_rules(self) -> Dict[str, List[str]]:
        """Load intervention synergy rules"""
        return {
            "zebra_crossing": [
                "speed_hump",
                "street_lighting",
                "pedestrian_bridge"
            ],
            "speed_limit_sign": [
                "speed_camera",
                "speed_hump",
                "enforcement"
            ],
            "traffic_light": [
                "pedestrian_crossing",
                "vehicle_detection",
                "priority_system"
            ],
            "school_zone_sign": [
                "speed_hump",
                "zebra_crossing",
                "flashing_lights"
            ]
        }
    
    def get_dependencies(self, intervention: str) -> List[str]:
        """Get required dependencies for intervention"""
        return self.dependency_rules.get(intervention, [])
    
    def get_conflicts(self, intervention: str) -> List[str]:
        """Get conflicting interventions"""
        return self.conflict_rules.get(intervention, [])
    
    def get_synergies(self, intervention: str) -> List[str]:
        """Get synergistic interventions"""
        return self.synergy_rules.get(intervention, [])
    
    def check_compliance(self, intervention: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check IRC/MoRTH compliance for intervention"""
        compliance = {
            "compliant": True,
            "violations": [],
            "recommendations": []
        }
        
        # Check dependencies
        dependencies = self.get_dependencies(intervention)
        for dep in dependencies:
            if dep not in context.get("interventions_present", []):
                compliance["violations"].append(f"Missing required dependency: {dep}")
                compliance["recommendations"].append(f"Install {dep} before {intervention}")
        
        # Check conflicts
        conflicts = self.get_conflicts(intervention)
        for conflict in conflicts:
            if conflict in context.get("interventions_present", []):
                compliance["violations"].append(f"Conflicts with existing intervention: {conflict}")
                compliance["recommendations"].append(f"Remove or relocate {conflict}")
        
        if compliance["violations"]:
            compliance["compliant"] = False
        
        return compliance

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for cascading effect prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GNN"""
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Attention mechanism
        x = self.attention(x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class HybridCascadingPredictor:
    """
    Hybrid system combining rule-based and ML-based cascading effect prediction
    """
    
    def __init__(self):
        self.rule_engine = RuleBasedDependencies()
        self.gnn_model = None
        self.road_network = nx.DiGraph()
        self.node_features = {}
        self.edge_features = {}
        
        # Feature encoders
        self.intervention_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        
        # Model parameters
        self.input_dim = 50  # Will be determined during training
        self.hidden_dim = 64
        self.output_dim = 32
        
        # Data storage paths
        self.models_path = Path("models/cascading_effects")
        self.data_path = Path("data/cascading_effects")
        
        # Create directories
        for path in [self.models_path, self.data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Load existing model
        self._load_existing_model()
        
        logger.info("Hybrid cascading effects predictor initialized")
    
    def predict_cascading_effects(self, intervention: str, road_context: Dict[str, Any]) -> CascadingEffect:
        """
        Predict cascading effects using hybrid approach
        """
        try:
            # Step 1: Rule-based prediction (high confidence)
            rule_effects = self._predict_rule_based(intervention, road_context)
            
            # Step 2: ML-based prediction (data-driven)
            ml_effects = self._predict_ml_based(intervention, road_context)
            
            # Step 3: Hybrid ensemble
            combined_effects = self._ensemble_predictions(rule_effects, ml_effects)
            
            # Create cascading effect object
            cascading_effect = CascadingEffect(
                effect_id=f"effect_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                primary_intervention=intervention,
                secondary_effects=combined_effects["secondary_effects"],
                dependencies=combined_effects["dependencies"],
                conflicts=combined_effects["conflicts"],
                synergies=combined_effects["synergies"],
                confidence=combined_effects["confidence"],
                prediction_method="hybrid",
                explanation=combined_effects["explanation"]
            )
            
            return cascading_effect
            
        except Exception as e:
            logger.error(f"Error predicting cascading effects: {e}")
            return self._create_fallback_effect(intervention, road_context)
    
    def _predict_rule_based(self, intervention: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based cascading effect prediction"""
        try:
            # Get rule-based dependencies
            dependencies = self.rule_engine.get_dependencies(intervention)
            conflicts = self.rule_engine.get_conflicts(intervention)
            synergies = self.rule_engine.get_synergies(intervention)
            
            # Check compliance
            compliance = self.rule_engine.check_compliance(intervention, context)
            
            # Generate secondary effects based on rules
            secondary_effects = []
            
            # Traffic flow effects
            if intervention in ["speed_hump", "traffic_light"]:
                secondary_effects.append({
                    "type": "traffic_flow",
                    "description": f"{intervention} will reduce traffic speed",
                    "magnitude": 0.3,
                    "confidence": 0.9,
                    "affected_area": "local"
                })
            
            # Safety effects
            if intervention in ["zebra_crossing", "pedestrian_bridge"]:
                secondary_effects.append({
                    "type": "safety",
                    "description": f"{intervention} will improve pedestrian safety",
                    "magnitude": 0.4,
                    "confidence": 0.85,
                    "affected_area": "local"
                })
            
            # Economic effects
            if intervention in ["speed_camera", "traffic_light"]:
                secondary_effects.append({
                    "type": "economic",
                    "description": f"{intervention} may generate revenue",
                    "magnitude": 0.2,
                    "confidence": 0.7,
                    "affected_area": "local"
                })
            
            return {
                "secondary_effects": secondary_effects,
                "dependencies": dependencies,
                "conflicts": conflicts,
                "synergies": synergies,
                "confidence": 0.8,
                "explanation": f"Rule-based prediction using IRC/MoRTH standards"
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based prediction: {e}")
            return {
                "secondary_effects": [],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "confidence": 0.3,
                "explanation": "Rule-based prediction failed"
            }
    
    def _predict_ml_based(self, intervention: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """ML-based cascading effect prediction"""
        try:
            if self.gnn_model is None:
                logger.warning("GNN model not available, using rule-based only")
                return {
                    "secondary_effects": [],
                    "dependencies": [],
                    "conflicts": [],
                    "synergies": [],
                    "confidence": 0.2,
                    "explanation": "ML model not available"
                }
            
            # Prepare input features
            input_features = self._prepare_ml_features(intervention, context)
            
            # Create graph data
            graph_data = self._create_graph_data(input_features)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.gnn_model(
                    graph_data.x,
                    graph_data.edge_index,
                    graph_data.batch
                )
            
            # Convert prediction to effects
            effects = self._interpret_ml_prediction(prediction, intervention)
            
            return {
                "secondary_effects": effects["secondary_effects"],
                "dependencies": effects["dependencies"],
                "conflicts": effects["conflicts"],
                "synergies": effects["synergies"],
                "confidence": effects["confidence"],
                "explanation": f"ML-based prediction using learned patterns"
            }
            
        except Exception as e:
            logger.error(f"Error in ML-based prediction: {e}")
            return {
                "secondary_effects": [],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "confidence": 0.2,
                "explanation": "ML prediction failed"
            }
    
    def _ensemble_predictions(self, rule_effects: Dict[str, Any], ml_effects: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule-based and ML predictions"""
        try:
            # Weighted combination (rules have higher weight for dependencies)
            rule_weight = 0.7
            ml_weight = 0.3
            
            # Combine secondary effects
            combined_effects = rule_effects["secondary_effects"].copy()
            
            # Add ML effects that don't conflict with rules
            for ml_effect in ml_effects["secondary_effects"]:
                # Check for conflicts with rule-based effects
                conflicts = False
                for rule_effect in rule_effects["secondary_effects"]:
                    if ml_effect["type"] == rule_effect["type"] and abs(ml_effect["magnitude"] - rule_effect["magnitude"]) > 0.3:
                        conflicts = True
                        break
                
                if not conflicts:
                    # Adjust confidence based on ML weight
                    ml_effect["confidence"] *= ml_weight
                    combined_effects.append(ml_effect)
            
            # Combine dependencies (rules take precedence)
            combined_dependencies = list(set(rule_effects["dependencies"] + ml_effects["dependencies"]))
            
            # Combine conflicts (rules take precedence)
            combined_conflicts = list(set(rule_effects["conflicts"] + ml_effects["conflicts"]))
            
            # Combine synergies
            combined_synergies = list(set(rule_effects["synergies"] + ml_effects["synergies"]))
            
            # Calculate combined confidence
            combined_confidence = (rule_effects["confidence"] * rule_weight + 
                                 ml_effects["confidence"] * ml_weight)
            
            # Create explanation
            explanation = f"Hybrid prediction combining rule-based ({rule_effects['confidence']:.2f}) and ML-based ({ml_effects['confidence']:.2f}) approaches"
            
            return {
                "secondary_effects": combined_effects,
                "dependencies": combined_dependencies,
                "conflicts": combined_conflicts,
                "synergies": combined_synergies,
                "confidence": combined_confidence,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return rule_effects  # Fallback to rule-based
    
    def _prepare_ml_features(self, intervention: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare features for ML model"""
        features = {
            "intervention_type": intervention,
            "location": context.get("location", {}),
            "road_type": context.get("road_type", "unknown"),
            "traffic_volume": context.get("traffic_volume", "medium"),
            "existing_interventions": context.get("interventions_present", []),
            "weather": context.get("weather", "clear"),
            "time_of_day": context.get("time_of_day", "day"),
            "accident_history": context.get("accident_history", [])
        }
        
        return features
    
    def _create_graph_data(self, features: Dict[str, Any]) -> Data:
        """Create graph data for GNN"""
        try:
            # Create nodes
            nodes = []
            node_features = []
            
            # Add intervention node
            intervention_id = "intervention_0"
            nodes.append(intervention_id)
            intervention_features = self._encode_intervention_features(features["intervention_type"])
            node_features.append(intervention_features)
            
            # Add location nodes
            location_id = "location_0"
            nodes.append(location_id)
            location_features = self._encode_location_features(features["location"])
            node_features.append(location_features)
            
            # Add existing intervention nodes
            for i, existing_intervention in enumerate(features["existing_interventions"]):
                node_id = f"existing_{i}"
                nodes.append(node_id)
                existing_features = self._encode_intervention_features(existing_intervention)
                node_features.append(existing_features)
            
            # Create edges
            edge_index = []
            edge_attr = []
            
            # Connect intervention to location
            edge_index.append([0, 1])  # intervention -> location
            edge_attr.append([1.0])  # strong connection
            
            # Connect intervention to existing interventions
            for i in range(len(features["existing_interventions"])):
                edge_index.append([0, 2 + i])  # intervention -> existing
                edge_attr.append([0.5])  # medium connection
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Create batch tensor
            batch = torch.zeros(len(nodes), dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            
        except Exception as e:
            logger.error(f"Error creating graph data: {e}")
            # Return minimal graph data
            x = torch.tensor([[0.0] * self.input_dim], dtype=torch.float)
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()
            batch = torch.tensor([0], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, batch=batch)
    
    def _encode_intervention_features(self, intervention: str) -> List[float]:
        """Encode intervention as feature vector"""
        try:
            # Simple one-hot encoding for common interventions
            intervention_types = [
                "zebra_crossing", "speed_limit_sign", "traffic_light", "speed_hump",
                "warning_sign", "barrier", "pedestrian_bridge", "advance_warning",
                "school_zone_sign", "hospital_zone_sign", "speed_camera", "rumble_strip"
            ]
            
            features = [0.0] * len(intervention_types)
            
            if intervention in intervention_types:
                idx = intervention_types.index(intervention)
                features[idx] = 1.0
            
            # Add additional features
            features.extend([
                1.0 if "speed" in intervention.lower() else 0.0,
                1.0 if "pedestrian" in intervention.lower() else 0.0,
                1.0 if "sign" in intervention.lower() else 0.0,
                1.0 if "crossing" in intervention.lower() else 0.0
            ])
            
            # Pad to input_dim
            while len(features) < self.input_dim:
                features.append(0.0)
            
            return features[:self.input_dim]
            
        except Exception as e:
            logger.error(f"Error encoding intervention features: {e}")
            return [0.0] * self.input_dim
    
    def _encode_location_features(self, location: Dict[str, Any]) -> List[float]:
        """Encode location as feature vector"""
        try:
            features = []
            
            # Coordinates
            features.append(location.get("lat", 0.0))
            features.append(location.get("lon", 0.0))
            
            # City/State encoding (simplified)
            city = location.get("city", "unknown")
            state = location.get("state", "unknown")
            
            features.append(hash(city) % 1000 / 1000.0)
            features.append(hash(state) % 1000 / 1000.0)
            
            # Pad to input_dim
            while len(features) < self.input_dim:
                features.append(0.0)
            
            return features[:self.input_dim]
            
        except Exception as e:
            logger.error(f"Error encoding location features: {e}")
            return [0.0] * self.input_dim
    
    def _interpret_ml_prediction(self, prediction: torch.Tensor, intervention: str) -> Dict[str, Any]:
        """Interpret ML model prediction"""
        try:
            # Convert prediction tensor to effects
            prediction_np = prediction.cpu().numpy()
            
            # Simple interpretation (would need more sophisticated approach)
            effects = {
                "secondary_effects": [],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "confidence": float(np.mean(prediction_np))
            }
            
            # Generate effects based on prediction values
            if prediction_np[0] > 0.5:
                effects["secondary_effects"].append({
                    "type": "traffic_impact",
                    "description": f"{intervention} predicted to impact traffic flow",
                    "magnitude": float(prediction_np[0]),
                    "confidence": 0.6,
                    "affected_area": "local"
                })
            
            if prediction_np[1] > 0.5:
                effects["secondary_effects"].append({
                    "type": "safety_impact",
                    "description": f"{intervention} predicted to improve safety",
                    "magnitude": float(prediction_np[1]),
                    "confidence": 0.6,
                    "affected_area": "local"
                })
            
            return effects
            
        except Exception as e:
            logger.error(f"Error interpreting ML prediction: {e}")
            return {
                "secondary_effects": [],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "confidence": 0.3
            }
    
    def _create_fallback_effect(self, intervention: str, context: Dict[str, Any]) -> CascadingEffect:
        """Create fallback cascading effect when prediction fails"""
        return CascadingEffect(
            effect_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            primary_intervention=intervention,
            secondary_effects=[{
                "type": "general",
                "description": f"{intervention} may have local traffic and safety impacts",
                "magnitude": 0.3,
                "confidence": 0.5,
                "affected_area": "local"
            }],
            dependencies=self.rule_engine.get_dependencies(intervention),
            conflicts=self.rule_engine.get_conflicts(intervention),
            synergies=self.rule_engine.get_synergies(intervention),
            confidence=0.4,
            prediction_method="fallback",
            explanation="Fallback prediction due to system error"
        )
    
    def _load_existing_model(self):
        """Load existing GNN model"""
        try:
            model_file = self.models_path / "gnn_model.pth"
            if model_file.exists():
                self.gnn_model = GraphNeuralNetwork(
                    self.input_dim, self.hidden_dim, self.output_dim
                )
                self.gnn_model.load_state_dict(torch.load(model_file))
                self.gnn_model.eval()
                logger.info("Loaded existing GNN model")
            else:
                logger.info("No existing GNN model found")
                
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
    
    def train_model(self, training_data: List[Dict[str, Any]]):
        """Train GNN model on historical data"""
        try:
            logger.info(f"Training GNN model on {len(training_data)} examples")
            
            # Initialize model
            self.gnn_model = GraphNeuralNetwork(
                self.input_dim, self.hidden_dim, self.output_dim
            )
            
            # Prepare training data
            train_loader = self._prepare_training_data(training_data)
            
            # Training loop (simplified)
            optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(10):  # Simplified training
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                    
                    # Compute loss (simplified)
                    target = torch.randn_like(output)  # Placeholder target
                    loss = criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")
            
            # Save trained model
            torch.save(self.gnn_model.state_dict(), self.models_path / "gnn_model.pth")
            logger.info("GNN model trained and saved")
            
        except Exception as e:
            logger.error(f"Error training GNN model: {e}")
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> DataLoader:
        """Prepare training data for GNN"""
        try:
            graph_data_list = []
            
            for data_point in training_data:
                # Create graph data for each training example
                graph_data = self._create_graph_data(data_point)
                graph_data_list.append(graph_data)
            
            # Create data loader
            loader = DataLoader(graph_data_list, batch_size=32, shuffle=True)
            return loader
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return DataLoader([], batch_size=32)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "rule_engine_rules": len(self.rule_engine.dependency_rules),
            "gnn_model_available": self.gnn_model is not None,
            "road_network_nodes": self.road_network.number_of_nodes(),
            "road_network_edges": self.road_network.number_of_edges(),
            "input_dimension": self.input_dim,
            "hidden_dimension": self.hidden_dim,
            "output_dimension": self.output_dim
        }

# Global instance
cascading_predictor = None

def get_cascading_predictor() -> HybridCascadingPredictor:
    """Get global cascading predictor instance"""
    global cascading_predictor
    if cascading_predictor is None:
        cascading_predictor = HybridCascadingPredictor()
    return cascading_predictor

def predict_cascading_effects(intervention: str, road_context: Dict[str, Any]) -> CascadingEffect:
    """Convenience function for cascading effect prediction"""
    predictor = get_cascading_predictor()
    return predictor.predict_cascading_effects(intervention, road_context)
