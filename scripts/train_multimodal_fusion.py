#!/usr/bin/env python3
"""
Multi-Modal Fusion Network Training
Trains network combining vision, text, accident data, and traffic patterns
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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """Dataset for multi-modal training data"""
    
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # Text features (already processed)
        text_features = torch.tensor(data['text_features'], dtype=torch.float32)
        
        # Image features (already processed)
        image_features = torch.tensor(data['image_features'], dtype=torch.float32)
        
        # Accident data features
        accident_features = torch.tensor(data['accident_features'], dtype=torch.float32)
        
        # Traffic pattern features
        traffic_features = torch.tensor(data['traffic_features'], dtype=torch.float32)
        
        # Target
        target = torch.tensor(data['target'], dtype=torch.float32)
        
        return {
            'text': text_features,
            'image': image_features,
            'accident': accident_features,
            'traffic': traffic_features,
            'target': target
        }

class VisionEncoder(nn.Module):
    """Vision encoder for road images"""
    
    def __init__(self, input_dim=512, hidden_dim=256):
        super(VisionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class TextEncoder(nn.Module):
    """Text encoder for road safety descriptions"""
    
    def __init__(self, input_dim=384, hidden_dim=256):
        super(TextEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class AccidentDataEncoder(nn.Module):
    """Encoder for accident data"""
    
    def __init__(self, input_dim=50, hidden_dim=128):
        super(AccidentDataEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class TrafficPatternEncoder(nn.Module):
    """Encoder for traffic pattern data"""
    
    def __init__(self, input_dim=30, hidden_dim=128):
        super(TrafficPatternEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, x):
        return self.encoder(x)

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.dim = dim
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.scale = dim ** -0.5
    
    def forward(self, x1, x2):
        # x1 queries x2
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        
        return out

class MultiModalFusionNetwork(nn.Module):
    """Multi-modal fusion network"""
    
    def __init__(self, 
                 text_dim=384, 
                 image_dim=512, 
                 accident_dim=50, 
                 traffic_dim=30,
                 hidden_dim=256,
                 output_dim=6):
        super(MultiModalFusionNetwork, self).__init__()
        
        # Individual encoders
        self.text_encoder = TextEncoder(text_dim, hidden_dim)
        self.image_encoder = VisionEncoder(image_dim, hidden_dim)
        self.accident_encoder = AccidentDataEncoder(accident_dim, hidden_dim)
        self.traffic_encoder = TrafficPatternEncoder(traffic_dim, hidden_dim)
        
        # Cross-modal attention
        self.text_image_attn = CrossModalAttention(hidden_dim // 2)
        self.image_text_attn = CrossModalAttention(hidden_dim // 2)
        self.accident_traffic_attn = CrossModalAttention(hidden_dim // 2)
        self.traffic_accident_attn = CrossModalAttention(hidden_dim // 2)
        
        # Fusion layers
        fusion_input_dim = (hidden_dim // 2) * 4  # 4 modalities
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Output heads
        self.intervention_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text, image, accident, traffic):
        # Encode each modality
        text_encoded = self.text_encoder(text)
        image_encoded = self.image_encoder(image)
        accident_encoded = self.accident_encoder(accident)
        traffic_encoded = self.traffic_encoder(traffic)
        
        # Cross-modal attention
        text_attended = self.text_image_attn(text_encoded, image_encoded)
        image_attended = self.image_text_attn(image_encoded, text_encoded)
        accident_attended = self.accident_traffic_attn(accident_encoded, traffic_encoded)
        traffic_attended = self.traffic_accident_attn(traffic_encoded, accident_encoded)
        
        # Concatenate all features
        fused_features = torch.cat([
            text_attended, image_attended, accident_attended, traffic_attended
        ], dim=-1)
        
        # Fusion
        fused = self.fusion(fused_features)
        
        # Output predictions
        intervention_pred = self.intervention_head(fused)
        confidence_pred = self.confidence_head(fused)
        
        return intervention_pred, confidence_pred

class MultiModalDataGenerator:
    """Generate multi-modal training data"""
    
    def __init__(self):
        self.interventions_db = self._load_interventions()
        self.accident_data = self._load_accident_data()
        
    def _load_interventions(self) -> List[Dict]:
        """Load intervention database"""
        try:
            with open("data/interventions/interventions_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load interventions: {e}")
            return []
    
    def _load_accident_data(self) -> pd.DataFrame:
        """Load accident data"""
        try:
            return pd.read_csv("data/accident_data/accident_dataset_100k.csv")
        except Exception as e:
            logger.error(f"Could not load accident data: {e}")
            return pd.DataFrame()
    
    def generate_training_data(self, num_samples: int = 50000) -> List[Dict]:
        """Generate multi-modal training data"""
        logger.info(f"Generating {num_samples} multi-modal training samples...")
        
        training_data = []
        
        for i in range(num_samples):
            # Select random intervention
            intervention = random.choice(self.interventions_db)
            
            # Generate text features (simulated embedding)
            text_features = self._generate_text_features(intervention)
            
            # Generate image features (simulated vision features)
            image_features = self._generate_image_features(intervention)
            
            # Generate accident data features
            accident_features = self._generate_accident_features()
            
            # Generate traffic pattern features
            traffic_features = self._generate_traffic_features()
            
            # Generate target (intervention effectiveness)
            target = self._generate_target(intervention, accident_features, traffic_features)
            
            training_data.append({
                'text_features': text_features,
                'image_features': image_features,
                'accident_features': accident_features,
                'traffic_features': traffic_features,
                'target': target,
                'intervention_id': intervention['intervention_id']
            })
        
        return training_data
    
    def _generate_text_features(self, intervention: Dict) -> List[float]:
        """Generate text features (simulated sentence transformer embedding)"""
        # Simulate 384-dimensional embedding
        features = np.random.randn(384)
        
        # Add some structure based on intervention properties
        cost = intervention['cost_estimate']['total']
        impact = intervention['predicted_impact']['accident_reduction_percent']
        
        # Modify features based on intervention characteristics
        features[0] = np.log10(cost + 1) / 10  # Cost feature
        features[1] = impact / 100  # Impact feature
        
        return features.tolist()
    
    def _generate_image_features(self, intervention: Dict) -> List[float]:
        """Generate image features (simulated vision features)"""
        # Simulate 512-dimensional vision features
        features = np.random.randn(512)
        
        # Add structure based on intervention type
        category = intervention.get('category', 'road_signs')
        category_map = {
            'road_signs': 0, 'road_markings': 1, 'traffic_calming': 2,
            'infrastructure': 3, 'pedestrian_facilities': 4, 'cyclist_facilities': 5,
            'smart_technology': 6
        }
        
        if category in category_map:
            features[category_map[category]] = 1.0
        
        return features.tolist()
    
    def _generate_accident_features(self) -> List[float]:
        """Generate accident data features"""
        if self.accident_data.empty:
            # Generate random features if no data available
            return np.random.randn(50).tolist()
        
        # Sample random accident record
        accident_record = self.accident_data.sample(1).iloc[0]
        
        features = [
            float(accident_record['speed_limit']) / 100,
            float(accident_record['actual_speed']) / 100,
            float(accident_record['road_width']) / 20,
            float(accident_record['lanes']) / 6,
            float(accident_record['num_interventions']) / 10,
            float(accident_record['severity_score']) / 20,
            float(accident_record['economic_impact']) / 10000000,
            float(accident_record['lives_lost']) / 10,
            float(accident_record['injuries']) / 20,
            float(accident_record['speed_violation']),
        ]
        
        # Add categorical features (one-hot encoded)
        weather_map = {'clear': 0, 'rain': 1, 'fog': 2, 'storm': 3, 'dust': 4}
        road_type_map = {'highway': 0, 'urban_arterial': 1, 'urban_collector': 2, 'rural_road': 3, 'city_street': 4}
        time_period_map = {'morning_rush': 0, 'midday': 1, 'evening_rush': 2, 'night': 3, 'late_night': 4}
        
        # Weather features
        weather_features = [0] * 5
        weather_features[weather_map.get(accident_record['weather'], 0)] = 1
        
        # Road type features
        road_type_features = [0] * 5
        road_type_features[road_type_map.get(accident_record['road_type'], 0)] = 1
        
        # Time period features
        time_period_features = [0] * 5
        time_period_features[time_period_map.get(accident_record['time_period'], 0)] = 1
        
        # Accident type features
        accident_type_features = [0] * 10
        accident_types = [
            'pedestrian_crossing', 'vehicle_collision', 'motorcycle_accident',
            'cyclist_accident', 'bus_accident', 'truck_accident', 'overtaking_accident',
            'lane_change_accident', 'signal_violation', 'speed_violation'
        ]
        if accident_record['accident_type'] in accident_types:
            accident_type_features[accident_types.index(accident_record['accident_type'])] = 1
        
        # Intervention features
        intervention_features = [0] * 9
        intervention_types = [
            'speed_hump', 'zebra_crossing', 'traffic_signal', 'speed_limit_sign',
            'warning_sign', 'barrier', 'street_light', 'reflector', 'rumble_strip'
        ]
        interventions_present = accident_record['interventions_present']
        for intervention in intervention_types:
            if intervention in interventions_present:
                intervention_features[intervention_types.index(intervention)] = 1
        
        features.extend(weather_features)
        features.extend(road_type_features)
        features.extend(time_period_features)
        features.extend(accident_type_features)
        features.extend(intervention_features)
        
        # Pad to expected size
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]
    
    def _generate_traffic_features(self) -> List[float]:
        """Generate traffic pattern features"""
        # Simulate traffic pattern data
        features = []
        
        # Hourly traffic volume (24 hours)
        for hour in range(24):
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                volume = float(random.uniform(0.7, 1.0))
            elif 22 <= hour or hour <= 5:  # Night hours
                volume = float(random.uniform(0.1, 0.3))
            else:  # Regular hours
                volume = float(random.uniform(0.4, 0.7))
            features.append(volume)
        
        # Additional traffic features
        features.extend([
            float(random.uniform(0.3, 0.9)),  # Average speed
            float(random.uniform(0.1, 0.8)),  # Congestion level
            float(random.uniform(0.0, 0.3)),  # Accident rate
            float(random.uniform(0.0, 0.2)),  # Violation rate
            float(random.uniform(0.0, 0.4)),  # Weather impact
        ])
        
        # Pad to expected size
        while len(features) < 30:
            features.append(0.0)
        
        return features[:30]
    
    def _generate_target(self, intervention: Dict, accident_features: List[float], 
                        traffic_features: List[float]) -> List[float]:
        """Generate target (intervention effectiveness)"""
        # Base effectiveness
        base_effectiveness = intervention['predicted_impact']['accident_reduction_percent'] / 100
        
        # Modify based on accident context
        accident_severity = accident_features[5] if len(accident_features) > 5 else 0.5
        accident_modifier = 1.0 + accident_severity * 0.2
        
        # Modify based on traffic context
        traffic_volume = np.mean(traffic_features[:24]) if len(traffic_features) >= 24 else 0.5
        traffic_modifier = 1.0 + traffic_volume * 0.3
        
        # Calculate final effectiveness
        effectiveness = base_effectiveness * accident_modifier * traffic_modifier
        effectiveness = min(effectiveness, 1.0)  # Cap at 100%
        
        # Generate multi-dimensional target
        target = [
            effectiveness,  # Overall effectiveness
            effectiveness * 0.8,  # Short-term effectiveness
            effectiveness * 1.2,  # Long-term effectiveness
            effectiveness * 0.6,  # Cost-effectiveness
            effectiveness * 0.9,  # Implementation feasibility
            effectiveness * 1.1,  # Maintenance sustainability
        ]
        
        return target

class MultiModalTrainer:
    """Trainer for multi-modal fusion network"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def train_model(self, training_data: List[Dict], epochs: int = 100, batch_size: int = 32):
        """Train multi-modal fusion model"""
        logger.info("Training multi-modal fusion network...")
        
        # Create dataset
        dataset = MultiModalDataset(training_data)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = MultiModalFusionNetwork().to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                accident = batch['accident'].to(self.device)
                traffic = batch['traffic'].to(self.device)
                target = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                
                intervention_pred, confidence_pred = self.model(text, image, accident, traffic)
                
                loss_intervention = criterion(intervention_pred, target)
                loss_confidence = criterion(confidence_pred, torch.ones_like(confidence_pred) * 0.8)
                
                total_loss = loss_intervention + 0.1 * loss_confidence
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    text = batch['text'].to(self.device)
                    image = batch['image'].to(self.device)
                    accident = batch['accident'].to(self.device)
                    traffic = batch['traffic'].to(self.device)
                    target = batch['target'].to(self.device)
                    
                    intervention_pred, confidence_pred = self.model(text, image, accident, traffic)
                    
                    loss_intervention = criterion(intervention_pred, target)
                    loss_confidence = criterion(confidence_pred, torch.ones_like(confidence_pred) * 0.8)
                    
                    total_loss = loss_intervention + 0.1 * loss_confidence
                    val_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/multimodal_fusion/fusion_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("Multi-modal fusion training completed!")
    
    def evaluate_model(self, training_data: List[Dict]):
        """Evaluate trained model"""
        logger.info("Evaluating multi-modal fusion network...")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/multimodal_fusion/fusion_model.pt'))
        self.model.eval()
        
        # Create test dataset
        test_size = int(0.2 * len(training_data))
        test_data = training_data[:test_size]
        test_dataset = MultiModalDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                text = batch['text'].to(self.device)
                image = batch['image'].to(self.device)
                accident = batch['accident'].to(self.device)
                traffic = batch['traffic'].to(self.device)
                target = batch['target'].to(self.device)
                
                intervention_pred, confidence_pred = self.model(text, image, accident, traffic)
                
                predictions.extend(intervention_pred.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        logger.info(f"Test MSE: {mse:.4f}")
        logger.info(f"Test R²: {r2:.4f}")
        
        # Save evaluation results
        results = {
            'mse': float(mse),
            'r2': float(r2),
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
        
        with open('models/multimodal_fusion/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

async def main():
    """Main function to train multi-modal fusion network"""
    logging.basicConfig(level=logging.INFO)
    
    # Create model directory
    Path("models/multimodal_fusion").mkdir(parents=True, exist_ok=True)
    
    # Generate training data
    generator = MultiModalDataGenerator()
    training_data = generator.generate_training_data(50000)
    
    # Save training data
    Path('data/multimodal').mkdir(parents=True, exist_ok=True)
    with open('data/multimodal/training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    logger.info("Multi-modal training data saved to data/multimodal/training_data.json")
    
    # Train model
    trainer = MultiModalTrainer()
    trainer.train_model(training_data, epochs=100)
    
    # Evaluate model
    trainer.evaluate_model(training_data)
    
    print("Multi-modal fusion network training completed successfully!")
    print("Model saved to: models/multimodal_fusion/fusion_model.pt")

if __name__ == "__main__":
    asyncio.run(main())
