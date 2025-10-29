#!/usr/bin/env python3
"""
Accident Prediction Ensemble Model Training
Trains Random Forest + Neural Network ensemble for accident prediction
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AccidentDataGenerator:
    """Generate comprehensive accident dataset for training"""
    
    def __init__(self):
        self.accident_types = [
            'pedestrian_crossing', 'vehicle_collision', 'motorcycle_accident',
            'cyclist_accident', 'bus_accident', 'truck_accident', 'overtaking_accident',
            'lane_change_accident', 'signal_violation', 'speed_violation'
        ]
        
        self.severity_levels = ['fatal', 'injury', 'property_damage']
        
        self.weather_conditions = ['clear', 'rain', 'fog', 'storm', 'dust']
        
        self.road_types = ['highway', 'urban_arterial', 'urban_collector', 'rural_road', 'city_street']
        
        self.time_periods = ['morning_rush', 'midday', 'evening_rush', 'night', 'late_night']
        
        self.intervention_types = [
            'speed_hump', 'zebra_crossing', 'traffic_signal', 'speed_limit_sign',
            'warning_sign', 'barrier', 'street_light', 'reflector', 'rumble_strip'
        ]
    
    def generate_accident_dataset(self, num_records: int = 100000) -> pd.DataFrame:
        """Generate comprehensive accident dataset"""
        logger.info(f"Generating {num_records} accident records...")
        
        records = []
        
        for i in range(num_records):
            # Basic accident info
            accident_type = random.choice(self.accident_types)
            severity = random.choice(self.severity_levels)
            weather = random.choice(self.weather_conditions)
            road_type = random.choice(self.road_types)
            time_period = random.choice(self.time_periods)
            
            # Location features
            latitude = random.uniform(8.0, 37.0)  # India latitude range
            longitude = random.uniform(68.0, 97.0)  # India longitude range
            
            # Traffic features
            traffic_volume = random.choice(['low', 'medium', 'high'])
            speed_limit = random.choice([30, 40, 50, 60, 80, 100])
            actual_speed = random.uniform(speed_limit * 0.8, speed_limit * 1.5)
            
            # Road features
            road_width = random.uniform(3.5, 12.0)  # meters
            lanes = random.randint(1, 6)
            road_condition = random.choice(['good', 'fair', 'poor'])
            
            # Intervention features
            interventions_present = random.sample(
                self.intervention_types, 
                random.randint(0, min(5, len(self.intervention_types)))
            )
            
            # Calculate accident probability based on features
            accident_probability = self._calculate_accident_probability(
                accident_type, severity, weather, road_type, time_period,
                traffic_volume, speed_limit, actual_speed, road_width,
                lanes, road_condition, interventions_present
            )
            
            # Generate accident outcome
            accident_occurred = random.random() < accident_probability
            
            if accident_occurred:
                # Accident severity score
                severity_score = self._calculate_severity_score(
                    severity, actual_speed, road_type, weather, interventions_present
                )
                
                # Economic impact
                economic_impact = self._calculate_economic_impact(severity, severity_score)
                
                # Lives affected
                lives_lost = self._calculate_lives_lost(severity, severity_score)
                injuries = self._calculate_injuries(severity, severity_score)
            else:
                severity_score = 0
                economic_impact = 0
                lives_lost = 0
                injuries = 0
            
            record = {
                'accident_id': f"ACC_{i:06d}",
                'accident_type': accident_type,
                'severity': severity,
                'severity_score': severity_score,
                'weather': weather,
                'road_type': road_type,
                'time_period': time_period,
                'latitude': latitude,
                'longitude': longitude,
                'traffic_volume': traffic_volume,
                'speed_limit': speed_limit,
                'actual_speed': actual_speed,
                'speed_violation': actual_speed > speed_limit,
                'road_width': road_width,
                'lanes': lanes,
                'road_condition': road_condition,
                'interventions_present': interventions_present,
                'num_interventions': len(interventions_present),
                'accident_occurred': accident_occurred,
                'economic_impact': economic_impact,
                'lives_lost': lives_lost,
                'injuries': injuries,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 365))
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated dataset with {len(df)} records")
        return df
    
    def _calculate_accident_probability(self, accident_type, severity, weather, road_type, 
                                     time_period, traffic_volume, speed_limit, actual_speed,
                                     road_width, lanes, road_condition, interventions_present) -> float:
        """Calculate accident probability based on features"""
        base_probability = 0.1  # Base 10% chance
        
        # Accident type modifiers
        type_modifiers = {
            'pedestrian_crossing': 0.15,
            'vehicle_collision': 0.12,
            'motorcycle_accident': 0.18,
            'cyclist_accident': 0.16,
            'bus_accident': 0.08,
            'truck_accident': 0.10,
            'overtaking_accident': 0.14,
            'lane_change_accident': 0.13,
            'signal_violation': 0.20,
            'speed_violation': 0.25
        }
        
        # Weather modifiers
        weather_modifiers = {
            'clear': 1.0,
            'rain': 1.5,
            'fog': 2.0,
            'storm': 2.5,
            'dust': 1.3
        }
        
        # Time period modifiers
        time_modifiers = {
            'morning_rush': 1.4,
            'midday': 1.0,
            'evening_rush': 1.6,
            'night': 1.2,
            'late_night': 0.8
        }
        
        # Traffic volume modifiers
        traffic_modifiers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.4
        }
        
        # Speed violation modifier
        speed_modifier = 1.0
        if actual_speed > speed_limit:
            speed_modifier = 1.0 + (actual_speed - speed_limit) / speed_limit * 0.5
        
        # Road condition modifier
        road_modifiers = {
            'good': 0.8,
            'fair': 1.0,
            'poor': 1.3
        }
        
        # Intervention modifier (safety interventions reduce accidents)
        intervention_modifier = 1.0 - len(interventions_present) * 0.05
        
        # Calculate final probability
        probability = (base_probability * 
                      type_modifiers.get(accident_type, 0.1) *
                      weather_modifiers.get(weather, 1.0) *
                      time_modifiers.get(time_period, 1.0) *
                      traffic_modifiers.get(traffic_volume, 1.0) *
                      speed_modifier *
                      road_modifiers.get(road_condition, 1.0) *
                      intervention_modifier)
        
        return min(probability, 0.8)  # Cap at 80%
    
    def _calculate_severity_score(self, severity, actual_speed, road_type, weather, interventions_present) -> float:
        """Calculate accident severity score"""
        base_score = {'fatal': 10, 'injury': 5, 'property_damage': 1}[severity]
        
        # Speed modifier
        speed_modifier = actual_speed / 50.0  # Normalize to 50 km/h
        
        # Weather modifier
        weather_modifiers = {'clear': 1.0, 'rain': 1.2, 'fog': 1.5, 'storm': 1.8, 'dust': 1.1}
        weather_modifier = weather_modifiers.get(weather, 1.0)
        
        # Intervention modifier (more interventions = lower severity)
        intervention_modifier = max(0.5, 1.0 - len(interventions_present) * 0.1)
        
        severity_score = base_score * speed_modifier * weather_modifier * intervention_modifier
        return min(severity_score, 20.0)  # Cap at 20
    
    def _calculate_economic_impact(self, severity, severity_score) -> float:
        """Calculate economic impact in INR"""
        base_costs = {
            'fatal': 5000000,  # 50 lakh
            'injury': 500000,  # 5 lakh
            'property_damage': 50000  # 50k
        }
        
        base_cost = base_costs.get(severity, 50000)
        return base_cost * (severity_score / 10.0)
    
    def _calculate_lives_lost(self, severity, severity_score) -> int:
        """Calculate number of lives lost"""
        if severity == 'fatal':
            return random.randint(1, max(1, int(severity_score / 5)))
        return 0
    
    def _calculate_injuries(self, severity, severity_score) -> int:
        """Calculate number of injuries"""
        if severity in ['fatal', 'injury']:
            return random.randint(1, max(1, int(severity_score / 3)))
        return 0

class AccidentPredictionNN(nn.Module):
    """Neural Network for accident prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(AccidentPredictionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layers for different predictions
        layers.append(nn.Linear(prev_dim, 1))  # Accident probability
        self.network = nn.Sequential(*layers)
        
        # Additional output heads
        self.severity_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.economic_impact_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.network[:-1](x)  # All layers except last
        accident_prob = torch.sigmoid(self.network[-1](features))
        severity = torch.relu(self.severity_head(features))
        economic_impact = torch.relu(self.economic_impact_head(features))
        
        return accident_prob, severity, economic_impact

class AccidentPredictionEnsemble:
    """Ensemble model combining Random Forest and Neural Network"""
    
    def __init__(self):
        self.rf_accident = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_severity = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_economic = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.nn_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare features for training"""
        logger.info("Preparing features...")
        
        # Create feature matrix
        features = []
        feature_names = []
        
        # Numerical features
        numerical_features = [
            'speed_limit', 'actual_speed', 'road_width', 'lanes', 'num_interventions'
        ]
        
        for feature in numerical_features:
            features.append(df[feature].values)
            feature_names.append(feature)
        
        # Categorical features (one-hot encoded)
        categorical_features = [
            'accident_type', 'weather', 'road_type', 'time_period', 
            'traffic_volume', 'road_condition'
        ]
        
        for feature in categorical_features:
            # One-hot encode
            dummies = pd.get_dummies(df[feature], prefix=feature)
            features.append(dummies.values)
            feature_names.extend(dummies.columns.tolist())
        
        # Binary features
        binary_features = ['speed_violation']
        for feature in binary_features:
            features.append(df[feature].astype(int).values)
            feature_names.append(feature)
        
        # Intervention features (count of each type)
        intervention_types = [
            'speed_hump', 'zebra_crossing', 'traffic_signal', 'speed_limit_sign',
            'warning_sign', 'barrier', 'street_light', 'reflector', 'rumble_strip'
        ]
        
        for intervention in intervention_types:
            count = df['interventions_present'].apply(
                lambda x: 1 if intervention in x else 0
            )
            features.append(count.values)
            feature_names.append(f'intervention_{intervention}')
        
        # Combine all features
        X = np.column_stack(features)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {feature_names}")
        
        return X, {'feature_names': feature_names}
    
    def train_ensemble(self, df: pd.DataFrame):
        """Train ensemble model"""
        logger.info("Training accident prediction ensemble...")
        
        # Prepare features
        X, feature_info = self.prepare_features(df)
        
        # Prepare targets
        y_accident = df['accident_occurred'].astype(int)
        y_severity = df['severity_score']
        y_economic = df['economic_impact']
        
        # Split data
        X_train, X_test, y_accident_train, y_accident_test = train_test_split(
            X, y_accident, test_size=0.2, random_state=42, stratify=y_accident
        )
        
        _, _, y_severity_train, y_severity_test = train_test_split(
            X, y_severity, test_size=0.2, random_state=42
        )
        
        _, _, y_economic_train, y_economic_test = train_test_split(
            X, y_economic, test_size=0.2, random_state=42
        )
        
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest models
        logger.info("Training Random Forest models...")
        
        self.rf_accident.fit(X_train, y_accident_train)
        self.rf_severity.fit(X_train, y_severity_train)
        self.rf_economic.fit(X_train, y_economic_train)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        
        input_dim = X_train_scaled.shape[1]
        self.nn_model = AccidentPredictionNN(input_dim).to(self.device)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_accident_train_tensor = torch.tensor(y_accident_train.values, dtype=torch.float32).to(self.device)
        y_severity_train_tensor = torch.tensor(y_severity_train.values, dtype=torch.float32).to(self.device)
        y_economic_train_tensor = torch.tensor(y_economic_train.values, dtype=torch.float32).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(200):
            self.nn_model.train()
            optimizer.zero_grad()
            
            accident_pred, severity_pred, economic_pred = self.nn_model(X_train_tensor)
            
            loss_accident = F.binary_cross_entropy(accident_pred.squeeze(), y_accident_train_tensor)
            loss_severity = criterion(severity_pred.squeeze(), y_severity_train_tensor)
            loss_economic = criterion(economic_pred.squeeze(), y_economic_train_tensor)
            
            total_loss = loss_accident + loss_severity + loss_economic
            
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
            
            # Validation
            if epoch % 20 == 0:
                self.nn_model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
                    val_accident, val_severity, val_economic = self.nn_model(X_test_tensor)
                    
                    val_loss_accident = F.binary_cross_entropy(val_accident.squeeze(), 
                                                             torch.tensor(y_accident_test.values, dtype=torch.float32).to(self.device))
                    val_loss_severity = criterion(val_severity.squeeze(), 
                                                torch.tensor(y_severity_test.values, dtype=torch.float32).to(self.device))
                    val_loss_economic = criterion(val_economic.squeeze(), 
                                                torch.tensor(y_economic_test.values, dtype=torch.float32).to(self.device))
                    
                    val_total_loss = val_loss_accident + val_loss_severity + val_loss_economic
                    
                    if val_total_loss.item() < best_loss:
                        best_loss = val_total_loss.item()
                        patience_counter = 0
                        torch.save(self.nn_model.state_dict(), 'models/accident_prediction/nn_model.pt')
                    else:
                        patience_counter += 1
                    
                    if epoch % 50 == 0:
                        logger.info(f"Epoch {epoch}: Train Loss: {total_loss.item():.4f}, Val Loss: {val_total_loss.item():.4f}")
            
            if patience_counter >= 30:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.nn_model.load_state_dict(torch.load('models/accident_prediction/nn_model.pt'))
        
        logger.info("Ensemble training completed!")
    
    def evaluate_ensemble(self, df: pd.DataFrame):
        """Evaluate ensemble model"""
        logger.info("Evaluating accident prediction ensemble...")
        
        # Prepare features
        X, feature_info = self.prepare_features(df)
        
        # Prepare targets
        y_accident = df['accident_occurred'].astype(int)
        y_severity = df['severity_score']
        y_economic = df['economic_impact']
        
        # Split data
        X_train, X_test, y_accident_train, y_accident_test = train_test_split(
            X, y_accident, test_size=0.2, random_state=42, stratify=y_accident
        )
        
        _, _, y_severity_train, y_severity_test = train_test_split(
            X, y_severity, test_size=0.2, random_state=42
        )
        
        _, _, y_economic_train, y_economic_test = train_test_split(
            X, y_economic, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest predictions
        rf_accident_pred = self.rf_accident.predict(X_test)
        rf_severity_pred = self.rf_severity.predict(X_test)
        rf_economic_pred = self.rf_economic.predict(X_test)
        
        # Neural Network predictions
        self.nn_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
            nn_accident_pred, nn_severity_pred, nn_economic_pred = self.nn_model(X_test_tensor)
            
            nn_accident_pred = nn_accident_pred.squeeze().cpu().numpy()
            nn_severity_pred = nn_severity_pred.squeeze().cpu().numpy()
            nn_economic_pred = nn_economic_pred.squeeze().cpu().numpy()
        
        # Ensemble predictions (weighted average)
        ensemble_accident_pred = 0.6 * rf_accident_pred + 0.4 * nn_accident_pred
        ensemble_severity_pred = 0.6 * rf_severity_pred + 0.4 * nn_severity_pred
        ensemble_economic_pred = 0.6 * rf_economic_pred + 0.4 * nn_economic_pred
        
        # Calculate metrics
        accident_accuracy = accuracy_score(y_accident_test, (ensemble_accident_pred > 0.5).astype(int))
        accident_precision = precision_score(y_accident_test, (ensemble_accident_pred > 0.5).astype(int))
        accident_recall = recall_score(y_accident_test, (ensemble_accident_pred > 0.5).astype(int))
        accident_f1 = f1_score(y_accident_test, (ensemble_accident_pred > 0.5).astype(int))
        
        severity_mse = mean_squared_error(y_severity_test, ensemble_severity_pred)
        severity_r2 = r2_score(y_severity_test, ensemble_severity_pred)
        
        economic_mse = mean_squared_error(y_economic_test, ensemble_economic_pred)
        economic_r2 = r2_score(y_economic_test, ensemble_economic_pred)
        
        # Log results
        logger.info("=== Ensemble Evaluation Results ===")
        logger.info(f"Accident Prediction:")
        logger.info(f"  Accuracy: {accident_accuracy:.4f}")
        logger.info(f"  Precision: {accident_precision:.4f}")
        logger.info(f"  Recall: {accident_recall:.4f}")
        logger.info(f"  F1-Score: {accident_f1:.4f}")
        
        logger.info(f"Severity Prediction:")
        logger.info(f"  MSE: {severity_mse:.4f}")
        logger.info(f"  R²: {severity_r2:.4f}")
        
        logger.info(f"Economic Impact Prediction:")
        logger.info(f"  MSE: {economic_mse:.4f}")
        logger.info(f"  R²: {economic_r2:.4f}")
        
        # Save results
        results = {
            'accident_prediction': {
                'accuracy': float(accident_accuracy),
                'precision': float(accident_precision),
                'recall': float(accident_recall),
                'f1_score': float(accident_f1)
            },
            'severity_prediction': {
                'mse': float(severity_mse),
                'r2': float(severity_r2)
            },
            'economic_prediction': {
                'mse': float(economic_mse),
                'r2': float(economic_r2)
            },
            'feature_names': feature_info['feature_names']
        }
        
        with open('models/accident_prediction/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving ensemble models...")
        
        # Save Random Forest models
        joblib.dump(self.rf_accident, 'models/accident_prediction/rf_accident.joblib')
        joblib.dump(self.rf_severity, 'models/accident_prediction/rf_severity.joblib')
        joblib.dump(self.rf_economic, 'models/accident_prediction/rf_economic.joblib')
        
        # Save scaler
        joblib.dump(self.scaler, 'models/accident_prediction/scaler.joblib')
        
        logger.info("Models saved successfully!")

async def main():
    """Main function to train accident prediction ensemble"""
    logging.basicConfig(level=logging.INFO)
    
    # Create model directory
    Path("models/accident_prediction").mkdir(parents=True, exist_ok=True)
    
    # Generate accident dataset
    generator = AccidentDataGenerator()
    df = generator.generate_accident_dataset(100000)
    
    # Save dataset
    df.to_csv('data/accident_data/accident_dataset_100k.csv', index=False)
    logger.info("Accident dataset saved to data/accident_data/accident_dataset_100k.csv")
    
    # Train ensemble model
    ensemble = AccidentPredictionEnsemble()
    ensemble.train_ensemble(df)
    
    # Evaluate model
    ensemble.evaluate_ensemble(df)
    
    # Save models
    ensemble.save_models()
    
    print("Accident prediction ensemble training completed successfully!")
    print("Models saved to: models/accident_prediction/")

if __name__ == "__main__":
    asyncio.run(main())
