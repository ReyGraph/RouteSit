import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class OptimizationResult:
    """Result of optimization process"""
    selected_interventions: List[str]
    total_cost: float
    total_impact: float
    cost_effectiveness: float
    implementation_timeline: int
    confidence_score: float
    pareto_score: float

class CostBenefitOptimizer:
    """Multi-objective optimization engine for intervention selection"""
    
    def __init__(self):
        self.weight_safety = 0.6
        self.weight_cost = 0.3
        self.weight_feasibility = 0.1
        self.max_scenarios = 5
    
    def optimize_interventions(self, 
                             interventions: List[Dict[str, Any]], 
                             budget_constraint: Optional[float] = None,
                             timeline_constraint: Optional[int] = None,
                             min_impact_threshold: Optional[float] = None) -> List[OptimizationResult]:
        """
        Optimize intervention selection using multi-objective optimization
        
        Args:
            interventions: List of candidate interventions
            budget_constraint: Maximum budget limit
            timeline_constraint: Maximum implementation timeline
            min_impact_threshold: Minimum required impact
            
        Returns:
            List of optimized intervention sets
        """
        try:
            logger.info(f"Optimizing {len(interventions)} interventions...")
            
            # Prepare optimization data
            optimization_data = self._prepare_optimization_data(interventions)
            
            # Generate multiple scenarios using different optimization strategies
            scenarios = []
            
            # Scenario 1: Maximum impact within budget
            if budget_constraint:
                scenarios.append(self._optimize_max_impact(optimization_data, budget_constraint))
            
            # Scenario 2: Minimum cost for target impact
            if min_impact_threshold:
                scenarios.append(self._optimize_min_cost(optimization_data, min_impact_threshold))
            
            # Scenario 3: Best cost-effectiveness ratio
            scenarios.append(self._optimize_cost_effectiveness(optimization_data))
            
            # Scenario 4: Pareto optimal solution
            scenarios.append(self._optimize_pareto(optimization_data))
            
            # Scenario 5: Balanced approach
            scenarios.append(self._optimize_balanced(optimization_data))
            
            # Filter and rank scenarios
            valid_scenarios = [s for s in scenarios if s is not None]
            ranked_scenarios = self._rank_scenarios(valid_scenarios)
            
            logger.info(f"Generated {len(ranked_scenarios)} optimization scenarios")
            
            return ranked_scenarios[:self.max_scenarios]
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return []
    
    def _prepare_optimization_data(self, interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for optimization"""
        data = {
            'interventions': interventions,
            'costs': np.array([intv['cost_estimate']['total'] for intv in interventions]),
            'impacts': np.array([intv['predicted_impact']['accident_reduction_percent'] for intv in interventions]),
            'timelines': np.array([intv['implementation_timeline'] for intv in interventions]),
            'confidence_levels': np.array([self._confidence_to_numeric(intv['predicted_impact']['confidence_level']) for intv in interventions]),
            'categories': [intv['category'] for intv in interventions]
        }
        
        # Calculate cost-effectiveness ratios
        data['cost_effectiveness'] = np.divide(data['impacts'], data['costs'], 
                                            out=np.zeros_like(data['impacts']), 
                                            where=data['costs']!=0)
        
        return data
    
    def _confidence_to_numeric(self, confidence: str) -> float:
        """Convert confidence level to numeric value"""
        confidence_map = {
            'high': 1.0,
            'medium': 0.7,
            'low': 0.4
        }
        return confidence_map.get(confidence.lower(), 0.5)
    
    def _optimize_max_impact(self, data: Dict[str, Any], budget_constraint: float) -> Optional[OptimizationResult]:
        """Optimize for maximum impact within budget constraint"""
        try:
            n_interventions = len(data['interventions'])
            
            # Objective function: maximize impact (minimize negative impact)
            def objective(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return 0
                return -np.sum(data['impacts'][selected_indices])
            
            # Constraint: budget limit
            def budget_constraint_func(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return budget_constraint
                return budget_constraint - np.sum(data['costs'][selected_indices])
            
            # Bounds: binary variables
            bounds = [(0, 1) for _ in range(n_interventions)]
            
            # Initial guess
            x0 = np.random.random(n_interventions)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': budget_constraint_func}
            )
            
            if result.success:
                selected_indices = np.where(result.x > 0.5)[0]
                return self._create_optimization_result(data, selected_indices, "Maximum Impact")
            
            return None
            
        except Exception as e:
            logger.error(f"Max impact optimization failed: {e}")
            return None
    
    def _optimize_min_cost(self, data: Dict[str, Any], min_impact_threshold: float) -> Optional[OptimizationResult]:
        """Optimize for minimum cost to achieve target impact"""
        try:
            n_interventions = len(data['interventions'])
            
            # Objective function: minimize cost
            def objective(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return float('inf')
                return np.sum(data['costs'][selected_indices])
            
            # Constraint: minimum impact threshold
            def impact_constraint_func(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return -min_impact_threshold
                return np.sum(data['impacts'][selected_indices]) - min_impact_threshold
            
            # Bounds: binary variables
            bounds = [(0, 1) for _ in range(n_interventions)]
            
            # Initial guess
            x0 = np.random.random(n_interventions)
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': impact_constraint_func}
            )
            
            if result.success:
                selected_indices = np.where(result.x > 0.5)[0]
                return self._create_optimization_result(data, selected_indices, "Minimum Cost")
            
            return None
            
        except Exception as e:
            logger.error(f"Min cost optimization failed: {e}")
            return None
    
    def _optimize_cost_effectiveness(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize for best cost-effectiveness ratio"""
        try:
            n_interventions = len(data['interventions'])
            
            # Objective function: maximize cost-effectiveness
            def objective(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return 0
                
                total_cost = np.sum(data['costs'][selected_indices])
                total_impact = np.sum(data['impacts'][selected_indices])
                
                if total_cost == 0:
                    return 0
                
                return -(total_impact / total_cost)  # Negative for minimization
            
            # Bounds: binary variables
            bounds = [(0, 1) for _ in range(n_interventions)]
            
            # Use differential evolution for global optimization
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100
            )
            
            if result.success:
                selected_indices = np.where(result.x > 0.5)[0]
                return self._create_optimization_result(data, selected_indices, "Cost-Effectiveness")
            
            return None
            
        except Exception as e:
            logger.error(f"Cost-effectiveness optimization failed: {e}")
            return None
    
    def _optimize_pareto(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Find Pareto optimal solution"""
        try:
            n_interventions = len(data['interventions'])
            
            # Multi-objective optimization: maximize impact, minimize cost
            def objective(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return [0, float('inf')]
                
                total_impact = np.sum(data['impacts'][selected_indices])
                total_cost = np.sum(data['costs'][selected_indices])
                
                # Normalize objectives
                max_impact = np.max(data['impacts'])
                max_cost = np.max(data['costs'])
                
                normalized_impact = total_impact / max_impact
                normalized_cost = total_cost / max_cost
                
                return [-normalized_impact, normalized_cost]  # Negative impact for minimization
            
            # Bounds: binary variables
            bounds = [(0, 1) for _ in range(n_interventions)]
            
            # Use weighted sum approach for Pareto optimization
            weights = [0.7, 0.3]  # Weight impact more than cost
            
            def weighted_objective(x):
                obj_values = objective(x)
                return weights[0] * obj_values[0] + weights[1] * obj_values[1]
            
            # Optimize
            result = differential_evolution(
                weighted_objective,
                bounds,
                seed=42,
                maxiter=100
            )
            
            if result.success:
                selected_indices = np.where(result.x > 0.5)[0]
                return self._create_optimization_result(data, selected_indices, "Pareto Optimal")
            
            return None
            
        except Exception as e:
            logger.error(f"Pareto optimization failed: {e}")
            return None
    
    def _optimize_balanced(self, data: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Balanced optimization considering all factors"""
        try:
            n_interventions = len(data['interventions'])
            
            # Multi-factor objective function
            def objective(x):
                selected_indices = np.where(x > 0.5)[0]
                if len(selected_indices) == 0:
                    return 0
                
                total_cost = np.sum(data['costs'][selected_indices])
                total_impact = np.sum(data['impacts'][selected_indices])
                avg_confidence = np.mean(data['confidence_levels'][selected_indices])
                max_timeline = np.max(data['timelines'][selected_indices])
                
                # Normalize factors
                max_cost = np.max(data['costs'])
                max_impact = np.max(data['impacts'])
                max_timeline_norm = np.max(data['timelines'])
                
                # Weighted score
                cost_score = total_cost / max_cost
                impact_score = total_impact / max_impact
                confidence_score = avg_confidence
                timeline_score = max_timeline / max_timeline_norm
                
                # Combined objective (minimize)
                combined_score = (
                    self.weight_cost * cost_score +
                    self.weight_safety * (1 - impact_score) +  # Negative impact
                    self.weight_feasibility * timeline_score
                ) - confidence_score  # Bonus for confidence
                
                return combined_score
            
            # Bounds: binary variables
            bounds = [(0, 1) for _ in range(n_interventions)]
            
            # Optimize
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100
            )
            
            if result.success:
                selected_indices = np.where(result.x > 0.5)[0]
                return self._create_optimization_result(data, selected_indices, "Balanced")
            
            return None
            
        except Exception as e:
            logger.error(f"Balanced optimization failed: {e}")
            return None
    
    def _create_optimization_result(self, data: Dict[str, Any], selected_indices: np.ndarray, strategy: str) -> OptimizationResult:
        """Create optimization result from selected indices"""
        if len(selected_indices) == 0:
            return None
        
        selected_interventions = [data['interventions'][i]['intervention_id'] for i in selected_indices]
        total_cost = np.sum(data['costs'][selected_indices])
        total_impact = np.sum(data['impacts'][selected_indices])
        cost_effectiveness = total_impact / total_cost if total_cost > 0 else 0
        implementation_timeline = int(np.max(data['timelines'][selected_indices]))
        confidence_score = float(np.mean(data['confidence_levels'][selected_indices]))
        
        # Calculate Pareto score (higher is better)
        pareto_score = (total_impact / np.max(data['impacts'])) - (total_cost / np.max(data['costs']))
        
        return OptimizationResult(
            selected_interventions=selected_interventions,
            total_cost=float(total_cost),
            total_impact=float(total_impact),
            cost_effectiveness=float(cost_effectiveness),
            implementation_timeline=implementation_timeline,
            confidence_score=confidence_score,
            pareto_score=float(pareto_score),
            strategy=strategy
        )
    
    def _rank_scenarios(self, scenarios: List[OptimizationResult]) -> List[OptimizationResult]:
        """Rank scenarios by overall quality"""
        if not scenarios:
            return []
        
        # Calculate ranking scores
        for scenario in scenarios:
            # Normalize scores
            max_cost = max(s.total_cost for s in scenarios)
            max_impact = max(s.total_impact for s in scenarios)
            max_effectiveness = max(s.cost_effectiveness for s in scenarios)
            
            # Weighted ranking score
            ranking_score = (
                0.4 * (scenario.total_impact / max_impact) +
                0.3 * (scenario.cost_effectiveness / max_effectiveness) +
                0.2 * scenario.confidence_score +
                0.1 * scenario.pareto_score
            )
            
            scenario.ranking_score = ranking_score
        
        # Sort by ranking score
        return sorted(scenarios, key=lambda x: x.ranking_score, reverse=True)
    
    def compare_scenarios(self, scenarios: List[OptimizationResult]) -> Dict[str, Any]:
        """Compare multiple optimization scenarios"""
        if not scenarios:
            return {}
        
        comparison = {
            'total_scenarios': len(scenarios),
            'best_cost_effective': min(scenarios, key=lambda x: x.total_cost / max(x.total_impact, 1)),
            'highest_impact': max(scenarios, key=lambda x: x.total_impact),
            'fastest_implementation': min(scenarios, key=lambda x: x.implementation_timeline),
            'most_confident': max(scenarios, key=lambda x: x.confidence_score),
            'cost_range': {
                'min': min(s.total_cost for s in scenarios),
                'max': max(s.total_cost for s in scenarios),
                'avg': np.mean([s.total_cost for s in scenarios])
            },
            'impact_range': {
                'min': min(s.total_impact for s in scenarios),
                'max': max(s.total_impact for s in scenarios),
                'avg': np.mean([s.total_impact for s in scenarios])
            }
        }
        
        return comparison
    
    def generate_recommendation_summary(self, scenarios: List[OptimizationResult]) -> str:
        """Generate human-readable recommendation summary"""
        if not scenarios:
            return "No optimization scenarios available."
        
        best_scenario = scenarios[0]  # Highest ranked
        
        summary_parts = []
        summary_parts.append("OPTIMIZATION RECOMMENDATION SUMMARY")
        summary_parts.append("=" * 50)
        
        summary_parts.append(f"\nRecommended Strategy: {best_scenario.strategy}")
        summary_parts.append(f"Number of Interventions: {len(best_scenario.selected_interventions)}")
        summary_parts.append(f"Total Cost: ₹{best_scenario.total_cost:,.0f}")
        summary_parts.append(f"Expected Impact: {best_scenario.total_impact:.1f}% accident reduction")
        summary_parts.append(f"Cost-Effectiveness: ₹{best_scenario.total_cost/max(best_scenario.total_impact, 1):,.0f} per percentage point")
        summary_parts.append(f"Implementation Timeline: {best_scenario.implementation_timeline} days")
        summary_parts.append(f"Confidence Level: {best_scenario.confidence_score:.1%}")
        
        summary_parts.append(f"\nAlternative Scenarios Available: {len(scenarios) - 1}")
        
        if len(scenarios) > 1:
            summary_parts.append("\nScenario Comparison:")
            for i, scenario in enumerate(scenarios[:3], 1):
                summary_parts.append(f"{i}. {scenario.strategy}: ₹{scenario.total_cost:,.0f}, {scenario.total_impact:.1f}% impact")
        
        return "\n".join(summary_parts)
