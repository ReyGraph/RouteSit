#!/usr/bin/env python3
"""
Pareto Optimization for Cost vs Lives vs Time Trade-offs
Implements multi-objective optimization with 3D visualization
"""

import os
import sys
import json
import logging
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ParetoOptimizer:
    """Pareto optimization for multi-objective road safety interventions"""
    
    def __init__(self):
        self.interventions_db = self._load_interventions()
        self.scaler = StandardScaler()
        
        # Objective weights (can be adjusted based on priorities)
        self.objective_weights = {
            'cost': 0.3,
            'lives_saved': 0.4,
            'implementation_time': 0.3
        }
        
        # Constraints
        self.constraints = {
            'max_cost': 10000000,  # 1 crore INR
            'min_lives_saved': 0.1,  # At least 0.1 lives saved per year
            'max_implementation_time': 365,  # 1 year max
            'min_effectiveness': 0.1  # At least 10% effectiveness
        }
    
    def _load_interventions(self) -> List[Dict]:
        """Load intervention database"""
        try:
            with open("data/interventions/interventions_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load interventions: {e}")
            return []
    
    def generate_intervention_scenarios(self, num_scenarios: int = 1000) -> List[Dict]:
        """Generate intervention scenarios for optimization"""
        logger.info(f"Generating {num_scenarios} intervention scenarios...")
        
        scenarios = []
        
        for i in range(num_scenarios):
            # Select random number of interventions (1-5)
            num_interventions = random.randint(1, 5)
            selected_interventions = random.sample(
                self.interventions_db, 
                min(num_interventions, len(self.interventions_db))
            )
            
            # Calculate scenario metrics
            scenario = self._calculate_scenario_metrics(selected_interventions)
            scenario['interventions'] = [intv['intervention_id'] for intv in selected_interventions]
            scenario['scenario_id'] = f"SCENARIO_{i:04d}"
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_scenario_metrics(self, interventions: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for intervention scenario"""
        total_cost = 0
        total_lives_saved = 0
        max_implementation_time = 0
        total_effectiveness = 0
        
        # Calculate synergies and conflicts
        synergy_bonus = 0
        conflict_penalty = 0
        
        for i, intervention1 in enumerate(interventions):
            # Basic metrics
            total_cost += intervention1['cost_estimate']['total']
            total_lives_saved += intervention1['predicted_impact']['lives_saved_per_year']
            max_implementation_time = max(
                max_implementation_time, 
                intervention1['implementation_timeline']['total'] if isinstance(intervention1['implementation_timeline'], dict) else intervention1['implementation_timeline']
            )
            total_effectiveness += intervention1['predicted_impact']['accident_reduction_percent']
            
            # Check for synergies and conflicts
            for intervention2 in interventions[i+1:]:
                if intervention1['intervention_id'] in intervention2.get('synergies', []):
                    synergy_bonus += 0.1  # 10% bonus per synergy
                if intervention1['intervention_id'] in intervention2.get('conflicts', []):
                    conflict_penalty += 0.15  # 15% penalty per conflict
        
        # Apply synergies and conflicts
        total_lives_saved *= (1 + synergy_bonus - conflict_penalty)
        total_effectiveness *= (1 + synergy_bonus - conflict_penalty)
        
        # Calculate composite metrics
        cost_per_life_saved = total_cost / max(total_lives_saved, 0.1)
        effectiveness_per_cost = total_effectiveness / max(total_cost, 1000)
        
        return {
            'total_cost': total_cost,
            'lives_saved_per_year': total_lives_saved,
            'implementation_time_days': max_implementation_time,
            'effectiveness_percent': min(total_effectiveness, 100),
            'cost_per_life_saved': cost_per_life_saved,
            'effectiveness_per_cost': effectiveness_per_cost,
            'synergy_bonus': synergy_bonus,
            'conflict_penalty': conflict_penalty,
            'num_interventions': len(interventions)
        }
    
    def find_pareto_frontier(self, scenarios: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """Find Pareto optimal solutions"""
        logger.info("Finding Pareto frontier...")
        
        # Extract objectives
        objectives = np.array([
            [scenario['total_cost'], 
             -scenario['lives_saved_per_year'],  # Negative because we want to maximize lives saved
             scenario['implementation_time_days']]
            for scenario in scenarios
        ])
        
        # Normalize objectives
        objectives_normalized = self.scaler.fit_transform(objectives)
        
        # Find Pareto optimal solutions
        pareto_indices = self._is_pareto_efficient(objectives_normalized)
        pareto_scenarios = [scenarios[i] for i in pareto_indices]
        pareto_objectives = objectives[pareto_indices]
        
        logger.info(f"Found {len(pareto_scenarios)} Pareto optimal solutions out of {len(scenarios)}")
        
        return pareto_scenarios, pareto_objectives
    
    def _is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        """Find Pareto efficient points"""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep points that are not dominated
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        
        return is_efficient
    
    def optimize_scenario(self, scenario: Dict, optimization_method: str = 'weighted_sum') -> Dict:
        """Optimize individual scenario"""
        logger.info(f"Optimizing scenario {scenario['scenario_id']}...")
        
        if optimization_method == 'weighted_sum':
            return self._weighted_sum_optimization(scenario)
        elif optimization_method == 'genetic_algorithm':
            return self._genetic_algorithm_optimization(scenario)
        elif optimization_method == 'pareto_ranking':
            return self._pareto_ranking_optimization(scenario)
        else:
            return scenario
    
    def _weighted_sum_optimization(self, scenario: Dict) -> Dict:
        """Weighted sum optimization"""
        # Normalize objectives
        cost_norm = scenario['total_cost'] / self.constraints['max_cost']
        lives_norm = scenario['lives_saved_per_year'] / 10  # Assume max 10 lives saved
        time_norm = scenario['implementation_time_days'] / self.constraints['max_implementation_time']
        
        # Calculate weighted score (lower is better)
        weighted_score = (
            self.objective_weights['cost'] * cost_norm +
            self.objective_weights['lives_saved'] * (1 - lives_norm) +  # Invert lives saved
            self.objective_weights['implementation_time'] * time_norm
        )
        
        scenario['optimization_score'] = weighted_score
        scenario['optimization_method'] = 'weighted_sum'
        
        return scenario
    
    def _genetic_algorithm_optimization(self, scenario: Dict) -> Dict:
        """Genetic algorithm optimization"""
        def objective_function(x):
            # x represents intervention selection probabilities
            cost = scenario['total_cost'] * x[0]
            lives = scenario['lives_saved_per_year'] * x[1]
            time = scenario['implementation_time_days'] * x[2]
            
            # Normalize and calculate score
            cost_norm = cost / self.constraints['max_cost']
            lives_norm = lives / 10
            time_norm = time / self.constraints['max_implementation_time']
            
            return (
                self.objective_weights['cost'] * cost_norm +
                self.objective_weights['lives_saved'] * (1 - lives_norm) +
                self.objective_weights['implementation_time'] * time_norm
            )
        
        # Define bounds for optimization variables
        bounds = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5)]
        
        # Run genetic algorithm
        result = differential_evolution(objective_function, bounds, seed=42)
        
        scenario['optimization_score'] = result.fun
        scenario['optimization_method'] = 'genetic_algorithm'
        scenario['optimization_result'] = {
            'success': result.success,
            'iterations': result.nit,
            'optimal_values': result.x.tolist()
        }
        
        return scenario
    
    def _pareto_ranking_optimization(self, scenario: Dict) -> Dict:
        """Pareto ranking optimization"""
        # Calculate dominance count
        dominance_count = 0
        
        # Compare with other scenarios (simplified)
        for other_scenario in self.interventions_db[:100]:  # Sample comparison
            if self._dominates(other_scenario, scenario):
                dominance_count += 1
        
        scenario['optimization_score'] = dominance_count
        scenario['optimization_method'] = 'pareto_ranking'
        
        return scenario
    
    def _dominates(self, scenario1: Dict, scenario2: Dict) -> bool:
        """Check if scenario1 dominates scenario2"""
        # Scenario1 dominates scenario2 if it's better in at least one objective
        # and not worse in any objective
        
        better_cost = scenario1['total_cost'] < scenario2['total_cost']
        better_lives = scenario1['lives_saved_per_year'] > scenario2['lives_saved_per_year']
        better_time = scenario1['implementation_time_days'] < scenario2['implementation_time_days']
        
        # At least one better and none worse
        return (better_cost or better_lives or better_time) and not (
            scenario1['total_cost'] > scenario2['total_cost'] or
            scenario1['lives_saved_per_year'] < scenario2['lives_saved_per_year'] or
            scenario1['implementation_time_days'] > scenario2['implementation_time_days']
        )
    
    def visualize_pareto_frontier(self, pareto_scenarios: List[Dict], 
                                 pareto_objectives: np.ndarray, 
                                 save_path: str = 'data/pareto_optimization/pareto_frontier.html'):
        """Create 3D visualization of Pareto frontier"""
        logger.info("Creating Pareto frontier visualization...")
        
        # Create 3D scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=pareto_objectives[:, 0],  # Cost
                y=pareto_objectives[:, 1],  # Lives saved (negative)
                z=pareto_objectives[:, 2],  # Implementation time
                mode='markers',
                marker=dict(
                    size=8,
                    color=pareto_objectives[:, 0],  # Color by cost
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Cost (INR)")
                ),
                text=[f"Scenario {i}: Cost={scenario['total_cost']:,.0f}, Lives={scenario['lives_saved_per_year']:.1f}, Time={scenario['implementation_time_days']} days"
                      for i, scenario in enumerate(pareto_scenarios)],
                hovertemplate='<b>%{text}</b><br>' +
                             'Cost: %{x:,.0f} INR<br>' +
                             'Lives Saved: %{y:.1f}<br>' +
                             'Time: %{z} days<extra></extra>',
                name='Pareto Optimal Solutions'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title='Pareto Frontier: Cost vs Lives Saved vs Implementation Time',
            scene=dict(
                xaxis_title='Total Cost (INR)',
                yaxis_title='Lives Saved per Year',
                zaxis_title='Implementation Time (Days)',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                zaxis=dict(showgrid=True)
            ),
            width=1000,
            height=800
        )
        
        # Save visualization
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
        logger.info(f"Pareto frontier visualization saved to {save_path}")
        
        return fig
    
    def create_trade_off_analysis(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Create comprehensive trade-off analysis"""
        logger.info("Creating trade-off analysis...")
        
        # Sort scenarios by different criteria
        scenarios_by_cost = sorted(scenarios, key=lambda x: x['total_cost'])
        scenarios_by_lives = sorted(scenarios, key=lambda x: x['lives_saved_per_year'], reverse=True)
        scenarios_by_time = sorted(scenarios, key=lambda x: x['implementation_time_days'])
        scenarios_by_effectiveness = sorted(scenarios, key=lambda x: x['effectiveness_percent'], reverse=True)
        
        # Calculate trade-off metrics
        trade_off_analysis = {
            'cost_lives_trade_off': self._calculate_trade_off_curve(scenarios, 'total_cost', 'lives_saved_per_year'),
            'cost_time_trade_off': self._calculate_trade_off_curve(scenarios, 'total_cost', 'implementation_time_days'),
            'lives_time_trade_off': self._calculate_trade_off_curve(scenarios, 'lives_saved_per_year', 'implementation_time_days'),
            'best_scenarios': {
                'lowest_cost': scenarios_by_cost[0],
                'highest_lives_saved': scenarios_by_lives[0],
                'fastest_implementation': scenarios_by_time[0],
                'highest_effectiveness': scenarios_by_effectiveness[0]
            },
            'statistics': {
                'total_scenarios': len(scenarios),
                'cost_range': (scenarios_by_cost[0]['total_cost'], scenarios_by_cost[-1]['total_cost']),
                'lives_range': (scenarios_by_lives[-1]['lives_saved_per_year'], scenarios_by_lives[0]['lives_saved_per_year']),
                'time_range': (scenarios_by_time[0]['implementation_time_days'], scenarios_by_time[-1]['implementation_time_days']),
                'effectiveness_range': (scenarios_by_effectiveness[-1]['effectiveness_percent'], scenarios_by_effectiveness[0]['effectiveness_percent'])
            }
        }
        
        return trade_off_analysis
    
    def _calculate_trade_off_curve(self, scenarios: List[Dict], x_metric: str, y_metric: str) -> Dict[str, List]:
        """Calculate trade-off curve between two metrics"""
        # Sort scenarios by x_metric
        sorted_scenarios = sorted(scenarios, key=lambda x: x[x_metric])
        
        x_values = [scenario[x_metric] for scenario in sorted_scenarios]
        y_values = [scenario[y_metric] for scenario in sorted_scenarios]
        
        return {
            'x_values': x_values,
            'y_values': y_values,
            'x_metric': x_metric,
            'y_metric': y_metric
        }
    
    def generate_recommendations(self, pareto_scenarios: List[Dict], 
                               user_preferences: Dict[str, float] = None) -> List[Dict]:
        """Generate recommendations based on Pareto frontier and user preferences"""
        logger.info("Generating recommendations...")
        
        if user_preferences is None:
            user_preferences = {
                'cost_weight': 0.3,
                'lives_weight': 0.4,
                'time_weight': 0.3
            }
        
        # Calculate recommendation scores
        recommendations = []
        
        for scenario in pareto_scenarios:
            # Normalize metrics
            cost_norm = scenario['total_cost'] / self.constraints['max_cost']
            lives_norm = scenario['lives_saved_per_year'] / 10
            time_norm = scenario['implementation_time_days'] / self.constraints['max_implementation_time']
            
            # Calculate weighted score
            score = (
                user_preferences['cost_weight'] * (1 - cost_norm) +  # Lower cost is better
                user_preferences['lives_weight'] * lives_norm +      # Higher lives saved is better
                user_preferences['time_weight'] * (1 - time_norm)    # Lower time is better
            )
            
            recommendation = {
                'scenario_id': scenario['scenario_id'],
                'score': score,
                'total_cost': scenario['total_cost'],
                'lives_saved_per_year': scenario['lives_saved_per_year'],
                'implementation_time_days': scenario['implementation_time_days'],
                'effectiveness_percent': scenario['effectiveness_percent'],
                'cost_per_life_saved': scenario['cost_per_life_saved'],
                'interventions': scenario['interventions'],
                'rationale': self._generate_recommendation_rationale(scenario, score)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _generate_recommendation_rationale(self, scenario: Dict, score: float) -> str:
        """Generate rationale for recommendation"""
        rationale_parts = []
        
        if scenario['cost_per_life_saved'] < 1000000:  # Less than 10 lakh per life
            rationale_parts.append("Highly cost-effective intervention")
        
        if scenario['lives_saved_per_year'] > 2:
            rationale_parts.append("Significant life-saving potential")
        
        if scenario['implementation_time_days'] < 30:
            rationale_parts.append("Quick implementation possible")
        
        if scenario['effectiveness_percent'] > 50:
            rationale_parts.append("High effectiveness in accident reduction")
        
        if scenario['synergy_bonus'] > 0:
            rationale_parts.append("Synergistic benefits with other interventions")
        
        if not rationale_parts:
            rationale_parts.append("Balanced approach with moderate benefits")
        
        return "; ".join(rationale_parts)

class ParetoOptimizationAPI:
    """API wrapper for Pareto optimization system"""
    
    def __init__(self):
        self.optimizer = ParetoOptimizer()
    
    async def optimize_interventions(self, 
                                   user_preferences: Dict[str, float] = None,
                                   num_scenarios: int = 1000) -> Dict[str, Any]:
        """Optimize interventions using Pareto analysis"""
        logger.info("Starting Pareto optimization...")
        
        # Generate scenarios
        scenarios = self.optimizer.generate_intervention_scenarios(num_scenarios)
        
        # Find Pareto frontier
        pareto_scenarios, pareto_objectives = self.optimizer.find_pareto_frontier(scenarios)
        
        # Create trade-off analysis
        trade_off_analysis = self.optimizer.create_trade_off_analysis(scenarios)
        
        # Generate recommendations
        recommendations = self.optimizer.generate_recommendations(pareto_scenarios, user_preferences)
        
        # Create visualization
        visualization_path = 'data/pareto_optimization/pareto_frontier.html'
        self.optimizer.visualize_pareto_frontier(pareto_scenarios, pareto_objectives, visualization_path)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(scenarios),
            'pareto_optimal_count': len(pareto_scenarios),
            'recommendations': recommendations,
            'trade_off_analysis': trade_off_analysis,
            'visualization_path': visualization_path,
            'user_preferences': user_preferences or self.optimizer.objective_weights
        }
        
        return results

async def main():
    """Test Pareto optimization system"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize API
    api = ParetoOptimizationAPI()
    
    # Test optimization
    user_preferences = {
        'cost_weight': 0.4,
        'lives_weight': 0.5,
        'time_weight': 0.1
    }
    
    print("Testing Pareto optimization system...")
    
    try:
        results = await api.optimize_interventions(user_preferences, num_scenarios=500)
        
        print("Optimization completed successfully!")
        print(f"Total scenarios analyzed: {results['total_scenarios']}")
        print(f"Pareto optimal solutions: {results['pareto_optimal_count']}")
        print(f"Top recommendation: {results['recommendations'][0]['scenario_id']}")
        print(f"Visualization saved to: {results['visualization_path']}")
        
        # Save results
        with open('data/pareto_optimization/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to data/pareto_optimization/optimization_results.json")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"Optimization failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
