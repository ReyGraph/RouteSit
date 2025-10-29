#!/usr/bin/env python3
"""
Scenario Generator with Pareto Optimization
Generates 3-5 intervention scenarios with cost vs effectiveness optimization
"""

import json
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import itertools

logger = logging.getLogger(__name__)

@dataclass
class InterventionScenario:
    """A complete intervention scenario"""
    scenario_name: str
    interventions: List[Dict[str, Any]]
    total_cost: int
    total_effectiveness: float
    implementation_time_days: int
    lives_saved_per_year: float
    roi_percentage: float
    confidence_level: float
    dependencies: List[str]
    conflicts: List[str]
    synergies: List[str]

@dataclass
class ParetoPoint:
    """A point on the Pareto frontier"""
    cost: int
    effectiveness: float
    scenario: InterventionScenario
    dominated: bool = False

class ScenarioGenerator:
    """Generates optimized intervention scenarios using Pareto optimization"""
    
    def __init__(self, intervention_db_path: str = "data/interventions_database.json"):
        self.intervention_db_path = Path(intervention_db_path)
        self.interventions = []
        self.scenario_templates = {
            "quick_fix": {
                "max_cost": 100000,
                "max_time": 7,
                "min_effectiveness": 0.2,
                "description": "Quick, low-cost interventions for immediate impact"
            },
            "medium_fix": {
                "max_cost": 500000,
                "max_time": 30,
                "min_effectiveness": 0.4,
                "description": "Balanced approach with moderate cost and good effectiveness"
            },
            "comprehensive": {
                "max_cost": 2000000,
                "max_time": 90,
                "min_effectiveness": 0.6,
                "description": "Comprehensive solution with maximum effectiveness"
            }
        }
        
        self._load_interventions()
    
    def _load_interventions(self):
        """Load intervention database"""
        if self.intervention_db_path.exists():
            try:
                with open(self.intervention_db_path, 'r') as f:
                    self.interventions = json.load(f)
                logger.info(f"Loaded {len(self.interventions)} interventions")
            except Exception as e:
                logger.error(f"Failed to load interventions: {e}")
                self._create_sample_interventions()
        else:
            logger.warning("Intervention database not found, creating sample data")
            self._create_sample_interventions()
    
    def _create_sample_interventions(self):
        """Create sample intervention data"""
        self.interventions = [
            {
                "intervention_id": "repaint_marking",
                "intervention_name": "Repaint Road Marking",
                "category": "road_marking",
                "cost_estimate": {"min": 10000, "max": 50000, "total": 30000},
                "effectiveness": 0.35,
                "implementation_time": {"min": 1, "max": 3, "total": 2},
                "complexity": "low",
                "dependencies": [],
                "conflicts": [],
                "synergies": ["street_lighting"]
            },
            {
                "intervention_id": "speed_hump",
                "intervention_name": "Install Speed Hump",
                "category": "traffic_calming",
                "cost_estimate": {"min": 15000, "max": 25000, "total": 20000},
                "effectiveness": 0.50,
                "implementation_time": {"min": 2, "max": 5, "total": 3},
                "complexity": "medium",
                "dependencies": ["warning_sign"],
                "conflicts": ["speed_camera"],
                "synergies": []
            },
            {
                "intervention_id": "warning_sign",
                "intervention_name": "Install Warning Sign",
                "category": "road_sign",
                "cost_estimate": {"min": 2000, "max": 5000, "total": 3500},
                "effectiveness": 0.25,
                "implementation_time": {"min": 1, "max": 2, "total": 1},
                "complexity": "low",
                "dependencies": [],
                "conflicts": [],
                "synergies": ["speed_camera"]
            },
            {
                "intervention_id": "street_lighting",
                "intervention_name": "Install Street Lighting",
                "category": "lighting",
                "cost_estimate": {"min": 50000, "max": 150000, "total": 100000},
                "effectiveness": 0.40,
                "implementation_time": {"min": 7, "max": 14, "total": 10},
                "complexity": "high",
                "dependencies": [],
                "conflicts": [],
                "synergies": ["road_marking"]
            },
            {
                "intervention_id": "speed_camera",
                "intervention_name": "Install Speed Camera",
                "category": "traffic_control",
                "cost_estimate": {"min": 80000, "max": 120000, "total": 100000},
                "effectiveness": 0.45,
                "implementation_time": {"min": 5, "max": 10, "total": 7},
                "complexity": "high",
                "dependencies": [],
                "conflicts": ["speed_hump"],
                "synergies": ["warning_sign"]
            },
            {
                "intervention_id": "pedestrian_crossing",
                "intervention_name": "Install Pedestrian Crossing",
                "category": "pedestrian_safety",
                "cost_estimate": {"min": 30000, "max": 80000, "total": 55000},
                "effectiveness": 0.60,
                "implementation_time": {"min": 3, "max": 7, "total": 5},
                "complexity": "medium",
                "dependencies": ["warning_sign"],
                "conflicts": [],
                "synergies": ["guard_rail"]
            },
            {
                "intervention_id": "guard_rail",
                "intervention_name": "Install Guard Rail",
                "category": "safety_barrier",
                "cost_estimate": {"min": 20000, "max": 60000, "total": 40000},
                "effectiveness": 0.30,
                "implementation_time": {"min": 2, "max": 5, "total": 3},
                "complexity": "medium",
                "dependencies": [],
                "conflicts": [],
                "synergies": ["pedestrian_crossing"]
            }
        ]
    
    def generate_scenarios(self, problem_context: Dict[str, Any], 
                          budget_constraint: Optional[int] = None,
                          time_constraint: Optional[int] = None) -> List[InterventionScenario]:
        """Generate multiple intervention scenarios"""
        
        # Filter interventions based on problem context
        relevant_interventions = self._filter_relevant_interventions(problem_context)
        
        if not relevant_interventions:
            logger.warning("No relevant interventions found")
            return []
        
        # Generate scenarios for each template
        scenarios = []
        
        for template_name, template_config in self.scenario_templates.items():
            # Apply constraints
            max_cost = budget_constraint or template_config["max_cost"]
            max_time = time_constraint or template_config["max_time"]
            
            scenario = self._generate_scenario_for_template(
                template_name, template_config, relevant_interventions, 
                max_cost, max_time, problem_context
            )
            
            if scenario:
                scenarios.append(scenario)
        
        # Generate additional optimized scenarios
        optimized_scenarios = self._generate_optimized_scenarios(
            relevant_interventions, budget_constraint, time_constraint, problem_context
        )
        
        scenarios.extend(optimized_scenarios)
        
        # Sort by cost-effectiveness ratio
        scenarios.sort(key=lambda x: x.lives_saved_per_year / (x.total_cost / 100000), reverse=True)
        
        return scenarios[:5]  # Return top 5 scenarios
    
    def _filter_relevant_interventions(self, problem_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter interventions based on problem context"""
        problem_text = problem_context.get("text_description", "").lower()
        road_type = problem_context.get("road_type", "").lower()
        traffic_volume = problem_context.get("traffic_volume", "").lower()
        
        relevant = []
        
        for intervention in self.interventions:
            intervention_name = intervention["intervention_name"].lower()
            category = intervention.get("category", "").lower()
            
            # Check if intervention matches problem keywords
            is_relevant = False
            
            if "faded" in problem_text or "marking" in problem_text:
                if "marking" in intervention_name or "crossing" in intervention_name:
                    is_relevant = True
            
            if "speed" in problem_text:
                if "speed" in intervention_name or "hump" in intervention_name:
                    is_relevant = True
            
            if "sign" in problem_text:
                if "sign" in intervention_name:
                    is_relevant = True
            
            if "light" in problem_text or "dark" in problem_text:
                if "light" in intervention_name:
                    is_relevant = True
            
            if "pedestrian" in problem_text:
                if "pedestrian" in intervention_name or "crossing" in intervention_name:
                    is_relevant = True
            
            # Always include some basic interventions
            if intervention["intervention_id"] in ["warning_sign", "repaint_marking"]:
                is_relevant = True
            
            if is_relevant:
                relevant.append(intervention)
        
        return relevant
    
    def _generate_scenario_for_template(self, template_name: str, template_config: Dict[str, Any],
                                      interventions: List[Dict[str, Any]], max_cost: int, max_time: int,
                                      problem_context: Dict[str, Any]) -> Optional[InterventionScenario]:
        """Generate scenario for a specific template"""
        
        selected_interventions = []
        total_cost = 0
        total_effectiveness = 0
        total_time = 0
        
        # Sort interventions by cost-effectiveness
        interventions_sorted = sorted(
            interventions, 
            key=lambda x: x["effectiveness"] / (x["cost_estimate"]["total"] / 10000),
            reverse=True
        )
        
        for intervention in interventions_sorted:
            intervention_cost = intervention["cost_estimate"]["total"]
            intervention_time = intervention["implementation_time"]["total"]
            
            # Check constraints
            if total_cost + intervention_cost <= max_cost and total_time + intervention_time <= max_time:
                selected_interventions.append(intervention)
                total_cost += intervention_cost
                total_effectiveness += intervention["effectiveness"]
                total_time += intervention_time
        
        if not selected_interventions:
            return None
        
        # Calculate synergies and conflicts
        dependencies, conflicts, synergies = self._analyze_intervention_interactions(selected_interventions)
        
        # Calculate lives saved and ROI
        lives_saved_per_year = total_effectiveness * 5.0  # Rough estimate
        roi_percentage = (lives_saved_per_year * 10000000) / total_cost * 100 if total_cost > 0 else 0
        
        return InterventionScenario(
            scenario_name=f"{template_name.replace('_', ' ').title()} Solution",
            interventions=selected_interventions,
            total_cost=total_cost,
            total_effectiveness=min(total_effectiveness, 1.0),
            implementation_time_days=total_time,
            lives_saved_per_year=lives_saved_per_year,
            roi_percentage=roi_percentage,
            confidence_level=0.8,
            dependencies=dependencies,
            conflicts=conflicts,
            synergies=synergies
        )
    
    def _generate_optimized_scenarios(self, interventions: List[Dict[str, Any]], 
                                    budget_constraint: Optional[int], time_constraint: Optional[int],
                                    problem_context: Dict[str, Any]) -> List[InterventionScenario]:
        """Generate Pareto-optimized scenarios"""
        
        # Generate all possible combinations (limited to avoid explosion)
        max_combinations = 3
        combinations = []
        
        for r in range(1, min(len(interventions) + 1, 4)):  # Max 3 interventions per scenario
            for combo in itertools.combinations(interventions, r):
                combinations.append(combo)
                if len(combinations) >= max_combinations:
                    break
            if len(combinations) >= max_combinations:
                break
        
        scenarios = []
        
        for combo in combinations:
            combo_list = list(combo)
            total_cost = sum(i["cost_estimate"]["total"] for i in combo_list)
            total_time = sum(i["implementation_time"]["total"] for i in combo_list)
            
            # Apply constraints
            if budget_constraint and total_cost > budget_constraint:
                continue
            if time_constraint and total_time > time_constraint:
                continue
            
            # Calculate effectiveness with synergies
            base_effectiveness = sum(i["effectiveness"] for i in combo_list)
            synergy_bonus = self._calculate_synergy_bonus(combo_list)
            total_effectiveness = min(base_effectiveness + synergy_bonus, 1.0)
            
            # Analyze interactions
            dependencies, conflicts, synergies = self._analyze_intervention_interactions(combo_list)
            
            # Skip if major conflicts
            if len(conflicts) > 0:
                continue
            
            lives_saved_per_year = total_effectiveness * 5.0
            roi_percentage = (lives_saved_per_year * 10000000) / total_cost * 100 if total_cost > 0 else 0
            
            scenario = InterventionScenario(
                scenario_name=f"Optimized Solution ({len(combo_list)} interventions)",
                interventions=combo_list,
                total_cost=total_cost,
                total_effectiveness=total_effectiveness,
                implementation_time_days=total_time,
                lives_saved_per_year=lives_saved_per_year,
                roi_percentage=roi_percentage,
                confidence_level=0.85,
                dependencies=dependencies,
                conflicts=conflicts,
                synergies=synergies
            )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_intervention_interactions(self, interventions: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
        """Analyze dependencies, conflicts, and synergies between interventions"""
        dependencies = []
        conflicts = []
        synergies = []
        
        intervention_ids = [i["intervention_id"] for i in interventions]
        
        for intervention in interventions:
            # Check dependencies
            for dep in intervention.get("dependencies", []):
                if dep not in intervention_ids:
                    dependencies.append(f"{intervention['intervention_name']} requires {dep}")
            
            # Check conflicts
            for conflict in intervention.get("conflicts", []):
                if conflict in intervention_ids:
                    conflicts.append(f"{intervention['intervention_name']} conflicts with {conflict}")
            
            # Check synergies
            for synergy in intervention.get("synergies", []):
                if synergy in intervention_ids:
                    synergies.append(f"{intervention['intervention_name']} synergizes with {synergy}")
        
        return dependencies, conflicts, synergies
    
    def _calculate_synergy_bonus(self, interventions: List[Dict[str, Any]]) -> float:
        """Calculate effectiveness bonus from synergies"""
        bonus = 0.0
        
        intervention_ids = [i["intervention_id"] for i in interventions]
        
        for intervention in interventions:
            for synergy in intervention.get("synergies", []):
                if synergy in intervention_ids:
                    bonus += 0.05  # 5% bonus per synergy
        
        return min(bonus, 0.2)  # Cap at 20% bonus
    
    def find_pareto_frontier(self, scenarios: List[InterventionScenario]) -> List[ParetoPoint]:
        """Find Pareto frontier for cost vs effectiveness"""
        points = []
        
        for scenario in scenarios:
            point = ParetoPoint(
                cost=scenario.total_cost,
                effectiveness=scenario.total_effectiveness,
                scenario=scenario
            )
            points.append(point)
        
        # Find non-dominated points
        pareto_points = []
        
        for i, point1 in enumerate(points):
            dominated = False
            
            for j, point2 in enumerate(points):
                if i != j:
                    # Check if point1 is dominated by point2
                    if (point2.cost <= point1.cost and point2.effectiveness >= point1.effectiveness and
                        (point2.cost < point1.cost or point2.effectiveness > point1.effectiveness)):
                        dominated = True
                        break
            
            if not dominated:
                pareto_points.append(point1)
        
        # Sort by cost
        pareto_points.sort(key=lambda x: x.cost)
        
        return pareto_points
    
    def generate_recommendations(self, scenarios: List[InterventionScenario], 
                               budget_constraint: Optional[int] = None) -> Dict[str, Any]:
        """Generate recommendations based on scenarios"""
        
        if not scenarios:
            return {"error": "No scenarios generated"}
        
        # Find best scenario for different criteria
        best_cost_effective = max(scenarios, key=lambda x: x.lives_saved_per_year / (x.total_cost / 100000))
        best_effectiveness = max(scenarios, key=lambda x: x.total_effectiveness)
        lowest_cost = min(scenarios, key=lambda x: x.total_cost)
        
        # Filter by budget if specified
        budget_filtered = [s for s in scenarios if s.total_cost <= budget_constraint] if budget_constraint else scenarios
        
        recommendations = {
            "best_overall": best_cost_effective.scenario_name,
            "best_effectiveness": best_effectiveness.scenario_name,
            "most_affordable": lowest_cost.scenario_name,
            "budget_options": [s.scenario_name for s in budget_filtered] if budget_constraint else [],
            "scenarios": []
        }
        
        # Add scenario details
        for scenario in scenarios[:3]:  # Top 3 scenarios
            recommendations["scenarios"].append({
                "name": scenario.scenario_name,
                "cost": f"Rs {scenario.total_cost:,}",
                "effectiveness": f"{scenario.total_effectiveness:.1%}",
                "time": f"{scenario.implementation_time_days} days",
                "lives_saved": f"{scenario.lives_saved_per_year:.1f} per year",
                "roi": f"{scenario.roi_percentage:.1f}%",
                "interventions": [i["intervention_name"] for i in scenario.interventions],
                "dependencies": scenario.dependencies,
                "conflicts": scenario.conflicts,
                "synergies": scenario.synergies
            })
        
        return recommendations

def main():
    """Test the scenario generator"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Scenario Generator...")
    
    generator = ScenarioGenerator()
    
    # Test problem context
    problem_context = {
        "text_description": "Faded zebra crossing at school zone intersection",
        "road_type": "Urban",
        "traffic_volume": "High",
        "image_analysis": {"detected_objects": ["zebra_crossing", "school_sign"]}
    }
    
    # Generate scenarios
    scenarios = generator.generate_scenarios(
        problem_context, 
        budget_constraint=200000,
        time_constraint=30
    )
    
    print(f"Generated {len(scenarios)} scenarios:")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario.scenario_name}")
        print(f"   Cost: Rs {scenario.total_cost:,}")
        print(f"   Effectiveness: {scenario.total_effectiveness:.1%}")
        print(f"   Time: {scenario.implementation_time_days} days")
        print(f"   Lives saved: {scenario.lives_saved_per_year:.1f} per year")
        print(f"   ROI: {scenario.roi_percentage:.1f}%")
        print(f"   Interventions: {[i['intervention_name'] for i in scenario.interventions]}")
    
    # Test Pareto frontier
    pareto_points = generator.find_pareto_frontier(scenarios)
    print(f"\nPareto frontier has {len(pareto_points)} points")
    
    # Test recommendations
    recommendations = generator.generate_recommendations(scenarios, budget_constraint=150000)
    print(f"\nRecommendations:")
    print(f"Best overall: {recommendations['best_overall']}")
    print(f"Most affordable: {recommendations['most_affordable']}")
    print(f"Budget options: {recommendations['budget_options']}")

if __name__ == "__main__":
    main()
