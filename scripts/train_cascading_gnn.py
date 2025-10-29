#!/usr/bin/env python3
"""
Graph Neural Network Training for Cascading Effects Prediction
Trains GNN to predict secondary effects of intervention combinations
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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from typing import Dict, List, Tuple, Any
import networkx as nx
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class InterventionGraphBuilder:
    """Build intervention dependency graph from database"""
    
    def __init__(self):
        self.interventions_db = self._load_interventions()
        self.graph = nx.DiGraph()
        self.node_features = {}
        self.edge_features = {}
        
    def _load_interventions(self) -> List[Dict]:
        """Load intervention database"""
        try:
            with open("data/interventions/interventions_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load interventions: {e}")
            return []
    
    def build_graph(self):
        """Build intervention dependency graph"""
        logger.info("Building intervention dependency graph...")
        
        # Add nodes (interventions)
        for intervention in self.interventions_db:
            node_id = intervention['intervention_id']
            self.graph.add_node(node_id)
            
            # Node features
            self.node_features[node_id] = self._extract_node_features(intervention)
        
        # Add edges (dependencies, conflicts, synergies)
        for intervention in self.interventions_db:
            node_id = intervention['intervention_id']
            
            # Dependencies (prerequisites)
            for dep in intervention.get('dependencies', []):
                if dep in self.node_features:
                    self.graph.add_edge(dep, node_id, relation='dependency')
            
            # Conflicts (incompatible)
            for conflict in intervention.get('conflicts', []):
                if conflict in self.node_features:
                    self.graph.add_edge(node_id, conflict, relation='conflict')
            
            # Synergies (complementary)
            for synergy in intervention.get('synergies', []):
                if synergy in self.node_features:
                    self.graph.add_edge(node_id, synergy, relation='synergy')
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _extract_node_features(self, intervention: Dict) -> List[float]:
        """Extract numerical features for intervention node"""
        features = []
        
        # Cost features
        cost = intervention['cost_estimate']['total']
        features.extend([
            np.log10(cost + 1),  # Log cost
            cost / 1000000,      # Normalized cost
        ])
        
        # Impact features
        impact = intervention['predicted_impact']
        features.extend([
            impact['accident_reduction_percent'] / 100,
            impact['lives_saved_per_year'],
            impact['injury_prevention_per_year'] / 100,
        ])
        
        # Timeline features
        timeline = intervention['implementation_timeline']
        if isinstance(timeline, dict):
            total_time = timeline.get('total', 30)
            features.extend([
                total_time / 100,  # Normalized timeline
                timeline.get('planning', 7) / total_time if total_time > 0 else 0,
                timeline.get('installation', 14) / total_time if total_time > 0 else 0,
            ])
        else:
            # Handle case where timeline is just a number
            features.extend([
                timeline / 100,  # Normalized timeline
                0.2,  # Default planning ratio
                0.6,  # Default installation ratio
            ])
        
        # Complexity features
        complexity = intervention.get('implementation_complexity', {})
        complexity_level = complexity.get('level', 'Medium')
        complexity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        features.append(complexity_map.get(complexity_level, 1))
        
        # Category features (one-hot encoded)
        category_map = {
            'road_signs': 0, 'road_markings': 1, 'traffic_calming': 2,
            'infrastructure': 3, 'pedestrian_facilities': 4, 'cyclist_facilities': 5,
            'smart_technology': 6
        }
        category_vector = [0] * 7
        category = intervention.get('category', 'road_signs')
        if category in category_map:
            category_vector[category_map[category]] = 1
        features.extend(category_vector)
        
        # Problem type features
        problem_map = {
            'damaged': 0, 'faded': 1, 'missing': 2, 'incorrect_placement': 3,
            'obstructed': 4, 'non_compliant': 5, 'ineffective': 6, 'outdated': 7,
            'insufficient': 8, 'poor_quality': 9, 'maintenance_required': 10, 'upgrade_needed': 11
        }
        problem_vector = [0] * 12
        problem_type = intervention.get('problem_type', 'damaged')
        if problem_type in problem_map:
            problem_vector[problem_map[problem_type]] = 1
        features.extend(problem_vector)
        
        return features
    
    def generate_training_data(self, num_samples: int = 10000) -> List[Dict]:
        """Generate training data for cascading effects"""
        logger.info(f"Generating {num_samples} training samples...")
        
        training_data = []
        
        for _ in range(num_samples):
            # Select random intervention combination
            num_interventions = random.randint(2, 5)
            selected_interventions = random.sample(
                list(self.node_features.keys()), 
                min(num_interventions, len(self.node_features))
            )
            
            # Calculate cascading effects
            cascading_effects = self._calculate_cascading_effects(selected_interventions)
            
            training_data.append({
                'interventions': selected_interventions,
                'cascading_effects': cascading_effects,
                'total_cost': sum(self.node_features[i][1] * 1000000 for i in selected_interventions),
                'total_impact': sum(self.node_features[i][3] for i in selected_interventions),
                'implementation_time': max(self.node_features[i][5] * 100 for i in selected_interventions)
            })
        
        return training_data
    
    def _calculate_cascading_effects(self, interventions: List[str]) -> Dict[str, float]:
        """Calculate cascading effects for intervention combination"""
        effects = {
            'accident_reduction_multiplier': 1.0,
            'cost_efficiency_multiplier': 1.0,
            'implementation_delay_multiplier': 1.0,
            'maintenance_burden_multiplier': 1.0,
            'synergy_bonus': 0.0,
            'conflict_penalty': 0.0
        }
        
        # Check for synergies
        synergy_count = 0
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                if self.graph.has_edge(intervention1, intervention2):
                    edge_data = self.graph[intervention1][intervention2]
                    if edge_data.get('relation') == 'synergy':
                        synergy_count += 1
        
        # Check for conflicts
        conflict_count = 0
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                if self.graph.has_edge(intervention1, intervention2):
                    edge_data = self.graph[intervention1][intervention2]
                    if edge_data.get('relation') == 'conflict':
                        conflict_count += 1
        
        # Calculate multipliers
        effects['synergy_bonus'] = synergy_count * 0.1  # 10% bonus per synergy
        effects['conflict_penalty'] = conflict_count * 0.15  # 15% penalty per conflict
        
        effects['accident_reduction_multiplier'] = 1.0 + effects['synergy_bonus'] - effects['conflict_penalty']
        effects['cost_efficiency_multiplier'] = 1.0 + effects['synergy_bonus'] * 0.5 - effects['conflict_penalty'] * 0.3
        
        # Implementation delays due to dependencies
        dependency_delays = 0
        for intervention in interventions:
            for pred in self.graph.predecessors(intervention):
                if pred not in interventions:
                    dependency_delays += 1
        
        effects['implementation_delay_multiplier'] = 1.0 + dependency_delays * 0.2
        
        return effects

class CascadingEffectsGNN(nn.Module):
    """Graph Neural Network for cascading effects prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 6):
        super(CascadingEffectsGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GNN"""
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Attention mechanism
        x = F.relu(self.attention(x, edge_index))
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CascadingEffectsTrainer:
    """Train GNN for cascading effects prediction"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.graph_builder = InterventionGraphBuilder()
        
    def prepare_data(self) -> Tuple[List[Data], List[Dict]]:
        """Prepare training data"""
        logger.info("Preparing training data...")
        
        # Build graph
        self.graph_builder.build_graph()
        
        # Generate training samples
        training_samples = self.graph_builder.generate_training_data(10000)
        
        # Convert to PyTorch Geometric format
        data_list = []
        targets = []
        
        for sample in training_samples:
            # Create subgraph for intervention combination
            subgraph_nodes = sample['interventions']
            
            # Node features
            node_features = []
            for node in subgraph_nodes:
                features = self.graph_builder.node_features[node]
                node_features.append(features)
            
            node_features = torch.tensor(node_features, dtype=torch.float)
            
            # Edge indices (create edges between all pairs)
            edge_indices = []
            for i, node1 in enumerate(subgraph_nodes):
                for j, node2 in enumerate(subgraph_nodes):
                    if i != j:
                        edge_indices.append([i, j])
            
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # Create data object
            data = Data(x=node_features, edge_index=edge_index)
            data_list.append(data)
            
            # Target (cascading effects)
            effects = sample['cascading_effects']
            target = torch.tensor([
                effects['accident_reduction_multiplier'],
                effects['cost_efficiency_multiplier'],
                effects['implementation_delay_multiplier'],
                effects['maintenance_burden_multiplier'],
                effects['synergy_bonus'],
                effects['conflict_penalty']
            ], dtype=torch.float)
            targets.append(target)
        
        logger.info(f"Prepared {len(data_list)} training samples")
        return data_list, targets
    
    def train_model(self, data_list: List[Data], targets: List[torch.Tensor], 
                   epochs: int = 100, batch_size: int = 32):
        """Train the GNN model"""
        logger.info("Training cascading effects GNN...")
        
        # Split data
        train_data, val_data, train_targets, val_targets = train_test_split(
            data_list, targets, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = data_list[0].x.shape[1]
        self.model = CascadingEffectsGNN(input_dim).to(self.device)
        
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
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                batch_targets = torch.stack(train_targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, batch_targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    batch_targets = torch.stack(val_targets[batch_idx * batch_size:(batch_idx + 1) * batch_size]).to(self.device)
                    
                    output = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(output, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/cascading_effects/gnn_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("GNN training completed!")
    
    def evaluate_model(self, data_list: List[Data], targets: List[torch.Tensor]):
        """Evaluate trained model"""
        logger.info("Evaluating cascading effects GNN...")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/cascading_effects/gnn_model.pt'))
        self.model.eval()
        
        # Create test loader
        test_loader = DataLoader(data_list, batch_size=32, shuffle=False)
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                batch_targets = torch.stack(targets[batch_idx * 32:(batch_idx + 1) * 32]).to(self.device)
                
                output = self.model(batch.x, batch.edge_index, batch.batch)
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())
        
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
        
        with open('models/cascading_effects/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

async def main():
    """Main function to train cascading effects GNN"""
    logging.basicConfig(level=logging.INFO)
    
    # Create model directory
    Path("models/cascading_effects").mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = CascadingEffectsTrainer()
    
    # Prepare data
    data_list, targets = trainer.prepare_data()
    
    # Train model
    trainer.train_model(data_list, targets, epochs=100)
    
    # Evaluate model
    trainer.evaluate_model(data_list, targets)
    
    print("Cascading effects GNN training completed successfully!")
    print("Model saved to: models/cascading_effects/gnn_model.pt")

if __name__ == "__main__":
    asyncio.run(main())
