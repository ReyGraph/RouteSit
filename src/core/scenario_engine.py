import json
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ScenarioRecommendation:
    """Scenario recommendation data structure"""
    scenario_name: str
    scenario_type: str  # 'quick_fix', 'medium_fix', 'comprehensive'
    interventions: List[Dict[str, Any]]
    total_cost: float
    timeline_days: int
    expected_impact_percent: float
    cost_effectiveness: float
    confidence_score: float
    dependencies: List[str]
    conflicts: List[str]
    synergies: List[str]
    implementation_phases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]

class ScenarioComparisonEngine:
    """Advanced scenario comparison engine with 3-tier recommendations"""
    
    def __init__(self):
        self.scenario_types = {
            'quick_fix': {
                'max_cost': 50000,
                'max_timeline': 7,
                'min_interventions': 1,
                'max_interventions': 3,
                'description': 'Immediate safety improvements with minimal disruption'
            },
            'medium_fix': {
                'max_cost': 200000,
                'max_timeline': 30,
                'min_interventions': 2,
                'max_interventions': 5,
                'description': 'Balanced approach with moderate investment'
            },
            'comprehensive': {
                'max_cost': 500000,
                'max_timeline': 90,
                'min_interventions': 3,
                'max_interventions': 8,
                'description': 'Complete safety transformation with maximum impact'
            }
        }
    
    def generate_scenarios(self, interventions: List[Dict[str, Any]], 
                          budget_constraint: float = None,
                          timeline_constraint: int = None,
                          impact_threshold: float = None) -> List[ScenarioRecommendation]:
        """
        Generate 3-tier scenario recommendations
        
        Args:
            interventions: List of candidate interventions
            budget_constraint: Maximum budget limit
            timeline_constraint: Maximum timeline
            impact_threshold: Minimum impact requirement
            
        Returns:
            List of scenario recommendations
        """
        try:
            logger.info(f"Generating scenarios for {len(interventions)} interventions")
            
            scenarios = []
            
            # Generate Quick Fix scenario
            quick_fix = self._generate_quick_fix_scenario(interventions, budget_constraint)
            if quick_fix:
                scenarios.append(quick_fix)
            
            # Generate Medium Fix scenario
            medium_fix = self._generate_medium_fix_scenario(interventions, budget_constraint)
            if medium_fix:
                scenarios.append(medium_fix)
            
            # Generate Comprehensive scenario
            comprehensive = self._generate_comprehensive_scenario(interventions, budget_constraint)
            if comprehensive:
                scenarios.append(comprehensive)
            
            # Apply constraints
            scenarios = self._apply_constraints(scenarios, budget_constraint, timeline_constraint, impact_threshold)
            
            # Rank scenarios
            scenarios = self._rank_scenarios(scenarios)
            
            logger.info(f"Generated {len(scenarios)} scenarios")
            return scenarios
            
        except Exception as e:
            logger.error(f"Scenario generation failed: {e}")
            return []
    
    def _generate_quick_fix_scenario(self, interventions: List[Dict[str, Any]], 
                                   budget_constraint: float = None) -> ScenarioRecommendation:
        """Generate quick fix scenario"""
        
        # Filter interventions for quick fix
        quick_interventions = [
            i for i in interventions 
            if (i['implementation_timeline'] <= 7 and 
                i['cost_estimate']['total'] <= 50000 and
                i['predicted_impact']['confidence_level'] in ['high', 'medium'])
        ]
        
        if not quick_interventions:
            return None
        
        # Sort by cost-effectiveness
        quick_interventions.sort(key=lambda x: x['predicted_impact']['accident_reduction_percent'] / max(x['cost_estimate']['total'], 1), reverse=True)
        
        # Select top interventions
        selected_interventions = quick_interventions[:3]
        
        # Calculate scenario metrics
        total_cost = sum(i['cost_estimate']['total'] for i in selected_interventions)
        timeline = max(i['implementation_timeline'] for i in selected_interventions)
        avg_impact = sum(i['predicted_impact']['accident_reduction_percent'] for i in selected_interventions) / len(selected_interventions)
        cost_effectiveness = avg_impact / max(total_cost / 1000, 1)
        
        # Calculate dependencies and conflicts
        dependencies = self._calculate_dependencies(selected_interventions)
        conflicts = self._calculate_conflicts(selected_interventions)
        synergies = self._calculate_synergies(selected_interventions)
        
        # Generate implementation phases
        implementation_phases = self._generate_implementation_phases(selected_interventions, 'quick_fix')
        
        # Risk assessment
        risk_assessment = self._assess_risks(selected_interventions, 'quick_fix')
        
        return ScenarioRecommendation(
            scenario_name="Quick Fix",
            scenario_type="quick_fix",
            interventions=selected_interventions,
            total_cost=total_cost,
            timeline_days=timeline,
            expected_impact_percent=avg_impact,
            cost_effectiveness=cost_effectiveness,
            confidence_score=0.8,
            dependencies=dependencies,
            conflicts=conflicts,
            synergies=synergies,
            implementation_phases=implementation_phases,
            risk_assessment=risk_assessment
        )
    
    def _generate_medium_fix_scenario(self, interventions: List[Dict[str, Any]], 
                                    budget_constraint: float = None) -> ScenarioRecommendation:
        """Generate medium fix scenario"""
        
        # Filter interventions for medium fix
        medium_interventions = [
            i for i in interventions 
            if (i['implementation_timeline'] <= 30 and 
                i['cost_estimate']['total'] <= 200000)
        ]
        
        if not medium_interventions:
            return None
        
        # Use optimization to select best combination
        selected_interventions = self._optimize_intervention_selection(
            medium_interventions, budget_constraint or 200000, max_interventions=5
        )
        
        if not selected_interventions:
            return None
        
        # Calculate scenario metrics
        total_cost = sum(i['cost_estimate']['total'] for i in selected_interventions)
        timeline = max(i['implementation_timeline'] for i in selected_interventions)
        avg_impact = sum(i['predicted_impact']['accident_reduction_percent'] for i in selected_interventions) / len(selected_interventions)
        cost_effectiveness = avg_impact / max(total_cost / 1000, 1)
        
        # Calculate dependencies and conflicts
        dependencies = self._calculate_dependencies(selected_interventions)
        conflicts = self._calculate_conflicts(selected_interventions)
        synergies = self._calculate_synergies(selected_interventions)
        
        # Generate implementation phases
        implementation_phases = self._generate_implementation_phases(selected_interventions, 'medium_fix')
        
        # Risk assessment
        risk_assessment = self._assess_risks(selected_interventions, 'medium_fix')
        
        return ScenarioRecommendation(
            scenario_name="Medium Fix",
            scenario_type="medium_fix",
            interventions=selected_interventions,
            total_cost=total_cost,
            timeline_days=timeline,
            expected_impact_percent=avg_impact,
            cost_effectiveness=cost_effectiveness,
            confidence_score=0.7,
            dependencies=dependencies,
            conflicts=conflicts,
            synergies=synergies,
            implementation_phases=implementation_phases,
            risk_assessment=risk_assessment
        )
    
    def _generate_comprehensive_scenario(self, interventions: List[Dict[str, Any]], 
                                       budget_constraint: float = None) -> ScenarioRecommendation:
        """Generate comprehensive scenario"""
        
        # Filter interventions for comprehensive fix
        comprehensive_interventions = [
            i for i in interventions 
            if i['cost_estimate']['total'] <= 500000
        ]
        
        if not comprehensive_interventions:
            return None
        
        # Use advanced optimization for comprehensive selection
        selected_interventions = self._optimize_intervention_selection(
            comprehensive_interventions, budget_constraint or 500000, max_interventions=8
        )
        
        if not selected_interventions:
            return None
        
        # Calculate scenario metrics
        total_cost = sum(i['cost_estimate']['total'] for i in selected_interventions)
        timeline = max(i['implementation_timeline'] for i in selected_interventions)
        avg_impact = sum(i['predicted_impact']['accident_reduction_percent'] for i in selected_interventions) / len(selected_interventions)
        cost_effectiveness = avg_impact / max(total_cost / 1000, 1)
        
        # Calculate dependencies and conflicts
        dependencies = self._calculate_dependencies(selected_interventions)
        conflicts = self._calculate_conflicts(selected_interventions)
        synergies = self._calculate_synergies(selected_interventions)
        
        # Generate implementation phases
        implementation_phases = self._generate_implementation_phases(selected_interventions, 'comprehensive')
        
        # Risk assessment
        risk_assessment = self._assess_risks(selected_interventions, 'comprehensive')
        
        return ScenarioRecommendation(
            scenario_name="Comprehensive Solution",
            scenario_type="comprehensive",
            interventions=selected_interventions,
            total_cost=total_cost,
            timeline_days=timeline,
            expected_impact_percent=avg_impact,
            cost_effectiveness=cost_effectiveness,
            confidence_score=0.9,
            dependencies=dependencies,
            conflicts=conflicts,
            synergies=synergies,
            implementation_phases=implementation_phases,
            risk_assessment=risk_assessment
        )
    
    def _optimize_intervention_selection(self, interventions: List[Dict[str, Any]], 
                                       max_budget: float, max_interventions: int) -> List[Dict[str, Any]]:
        """Optimize intervention selection using knapsack-like algorithm"""
        
        # Sort by cost-effectiveness
        interventions.sort(key=lambda x: x['predicted_impact']['accident_reduction_percent'] / max(x['cost_estimate']['total'], 1), reverse=True)
        
        selected = []
        total_cost = 0
        
        for intervention in interventions:
            if (len(selected) < max_interventions and 
                total_cost + intervention['cost_estimate']['total'] <= max_budget):
                selected.append(intervention)
                total_cost += intervention['cost_estimate']['total']
        
        return selected
    
    def _calculate_dependencies(self, interventions: List[Dict[str, Any]]) -> List[str]:
        """Calculate dependencies for selected interventions"""
        
        dependencies = []
        
        for intervention in interventions:
            intervention_deps = intervention.get('dependencies', [])
            for dep in intervention_deps:
                if dep not in dependencies:
                    dependencies.append(dep)
        
        return dependencies
    
    def _calculate_conflicts(self, interventions: List[Dict[str, Any]]) -> List[str]:
        """Calculate conflicts between selected interventions"""
        
        conflicts = []
        
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                intervention1_conflicts = intervention1.get('conflicts', [])
                intervention2_conflicts = intervention2.get('conflicts', [])
                
                # Check if interventions conflict with each other
                if (intervention2['intervention_name'] in intervention1_conflicts or
                    intervention1['intervention_name'] in intervention2_conflicts):
                    conflicts.append(f"{intervention1['intervention_name']} conflicts with {intervention2['intervention_name']}")
        
        return conflicts
    
    def _calculate_synergies(self, interventions: List[Dict[str, Any]]) -> List[str]:
        """Calculate synergies between selected interventions"""
        
        synergies = []
        
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                intervention1_synergies = intervention1.get('synergies', [])
                intervention2_synergies = intervention2.get('synergies', [])
                
                # Check if interventions synergize with each other
                if (intervention2['intervention_name'] in intervention1_synergies or
                    intervention1['intervention_name'] in intervention2_synergies):
                    synergies.append(f"{intervention1['intervention_name']} synergizes with {intervention2['intervention_name']}")
        
        return synergies
    
    def _generate_implementation_phases(self, interventions: List[Dict[str, Any]], 
                                      scenario_type: str) -> List[Dict[str, Any]]:
        """Generate implementation phases for the scenario"""
        
        phases = []
        
        if scenario_type == 'quick_fix':
            phases = [
                {
                    'phase': 'Phase 1: Immediate Actions',
                    'timeline': 'Days 1-3',
                    'interventions': [i for i in interventions if i['implementation_timeline'] <= 3],
                    'description': 'Quick repairs and basic improvements'
                }
            ]
        
        elif scenario_type == 'medium_fix':
            phases = [
                {
                    'phase': 'Phase 1: Immediate Actions',
                    'timeline': 'Days 1-7',
                    'interventions': [i for i in interventions if i['implementation_timeline'] <= 7],
                    'description': 'Quick repairs and basic improvements'
                },
                {
                    'phase': 'Phase 2: Short-term Improvements',
                    'timeline': 'Days 8-30',
                    'interventions': [i for i in interventions if 7 < i['implementation_timeline'] <= 30],
                    'description': 'Medium-term safety enhancements'
                }
            ]
        
        elif scenario_type == 'comprehensive':
            phases = [
                {
                    'phase': 'Phase 1: Immediate Actions',
                    'timeline': 'Days 1-7',
                    'interventions': [i for i in interventions if i['implementation_timeline'] <= 7],
                    'description': 'Quick repairs and basic improvements'
                },
                {
                    'phase': 'Phase 2: Short-term Improvements',
                    'timeline': 'Days 8-30',
                    'interventions': [i for i in interventions if 7 < i['implementation_timeline'] <= 30],
                    'description': 'Medium-term safety enhancements'
                },
                {
                    'phase': 'Phase 3: Long-term Infrastructure',
                    'timeline': 'Days 31-90',
                    'interventions': [i for i in interventions if i['implementation_timeline'] > 30],
                    'description': 'Major infrastructure improvements'
                }
            ]
        
        return phases
    
    def _assess_risks(self, interventions: List[Dict[str, Any]], scenario_type: str) -> Dict[str, Any]:
        """Assess risks for the scenario"""
        
        risks = {
            'implementation_risk': 'low',
            'budget_risk': 'low',
            'timeline_risk': 'low',
            'impact_risk': 'low',
            'overall_risk': 'low'
        }
        
        total_cost = sum(i['cost_estimate']['total'] for i in interventions)
        max_timeline = max(i['implementation_timeline'] for i in interventions)
        
        # Assess budget risk
        if total_cost > 300000:
            risks['budget_risk'] = 'high'
        elif total_cost > 150000:
            risks['budget_risk'] = 'medium'
        
        # Assess timeline risk
        if max_timeline > 60:
            risks['timeline_risk'] = 'high'
        elif max_timeline > 30:
            risks['timeline_risk'] = 'medium'
        
        # Assess implementation risk based on complexity
        complex_interventions = [i for i in interventions if i['implementation_timeline'] > 14]
        if len(complex_interventions) > 3:
            risks['implementation_risk'] = 'high'
        elif len(complex_interventions) > 1:
            risks['implementation_risk'] = 'medium'
        
        # Calculate overall risk
        risk_scores = {'low': 1, 'medium': 2, 'high': 3}
        overall_score = sum(risk_scores[risk] for risk in risks.values()) / len(risks)
        
        if overall_score >= 2.5:
            risks['overall_risk'] = 'high'
        elif overall_score >= 1.5:
            risks['overall_risk'] = 'medium'
        
        return risks
    
    def _apply_constraints(self, scenarios: List[ScenarioRecommendation], 
                          budget_constraint: float = None,
                          timeline_constraint: int = None,
                          impact_threshold: float = None) -> List[ScenarioRecommendation]:
        """Apply constraints to scenarios"""
        
        filtered_scenarios = []
        
        for scenario in scenarios:
            include_scenario = True
            
            if budget_constraint and scenario.total_cost > budget_constraint:
                include_scenario = False
            
            if timeline_constraint and scenario.timeline_days > timeline_constraint:
                include_scenario = False
            
            if impact_threshold and scenario.expected_impact_percent < impact_threshold:
                include_scenario = False
            
            if include_scenario:
                filtered_scenarios.append(scenario)
        
        return filtered_scenarios
    
    def _rank_scenarios(self, scenarios: List[ScenarioRecommendation]) -> List[ScenarioRecommendation]:
        """Rank scenarios by overall quality"""
        
        for scenario in scenarios:
            # Calculate ranking score
            cost_score = 1 - (scenario.total_cost / 500000)  # Normalize cost
            impact_score = scenario.expected_impact_percent / 100  # Normalize impact
            effectiveness_score = scenario.cost_effectiveness / 10  # Normalize effectiveness
            confidence_score = scenario.confidence_score
            
            # Weighted ranking
            ranking_score = (
                0.3 * cost_score +
                0.3 * impact_score +
                0.2 * effectiveness_score +
                0.2 * confidence_score
            )
            
            scenario.ranking_score = ranking_score
        
        # Sort by ranking score
        return sorted(scenarios, key=lambda x: x.ranking_score, reverse=True)
    
    def compare_scenarios(self, scenarios: List[ScenarioRecommendation]) -> Dict[str, Any]:
        """Compare multiple scenarios"""
        
        if not scenarios:
            return {}
        
        comparison = {
            'total_scenarios': len(scenarios),
            'scenario_summary': [],
            'cost_comparison': {
                'min_cost': min(s.total_cost for s in scenarios),
                'max_cost': max(s.total_cost for s in scenarios),
                'avg_cost': np.mean([s.total_cost for s in scenarios])
            },
            'impact_comparison': {
                'min_impact': min(s.expected_impact_percent for s in scenarios),
                'max_impact': max(s.expected_impact_percent for s in scenarios),
                'avg_impact': np.mean([s.expected_impact_percent for s in scenarios])
            },
            'timeline_comparison': {
                'min_timeline': min(s.timeline_days for s in scenarios),
                'max_timeline': max(s.timeline_days for s in scenarios),
                'avg_timeline': np.mean([s.timeline_days for s in scenarios])
            },
            'recommendations': {
                'best_cost_effective': min(scenarios, key=lambda x: x.total_cost / max(x.expected_impact_percent, 1)),
                'highest_impact': max(scenarios, key=lambda x: x.expected_impact_percent),
                'fastest_implementation': min(scenarios, key=lambda x: x.timeline_days),
                'most_comprehensive': max(scenarios, key=lambda x: len(x.interventions))
            }
        }
        
        # Generate scenario summaries
        for scenario in scenarios:
            comparison['scenario_summary'].append({
                'name': scenario.scenario_name,
                'type': scenario.scenario_type,
                'cost': scenario.total_cost,
                'timeline': scenario.timeline_days,
                'impact': scenario.expected_impact_percent,
                'interventions_count': len(scenario.interventions),
                'ranking_score': scenario.ranking_score
            })
        
        return comparison
    
    def generate_scenario_report(self, scenarios: List[ScenarioRecommendation]) -> str:
        """Generate human-readable scenario report"""
        
        if not scenarios:
            return "No scenarios available for comparison."
        
        report_parts = []
        report_parts.append("SCENARIO COMPARISON REPORT")
        report_parts.append("=" * 50)
        
        # Summary
        report_parts.append(f"\nTotal Scenarios: {len(scenarios)}")
        
        # Individual scenario details
        for i, scenario in enumerate(scenarios, 1):
            report_parts.append(f"\n{i}. {scenario.scenario_name}")
            report_parts.append(f"   Type: {scenario.scenario_type}")
            report_parts.append(f"   Cost: ₹{scenario.total_cost:,.0f}")
            report_parts.append(f"   Timeline: {scenario.timeline_days} days")
            report_parts.append(f"   Expected Impact: {scenario.expected_impact_percent:.1f}%")
            report_parts.append(f"   Interventions: {len(scenario.interventions)}")
            report_parts.append(f"   Cost-Effectiveness: ₹{scenario.total_cost/max(scenario.expected_impact_percent, 1):,.0f} per percentage point")
            
            # Dependencies
            if scenario.dependencies:
                report_parts.append(f"   Dependencies: {', '.join(scenario.dependencies[:3])}")
            
            # Conflicts
            if scenario.conflicts:
                report_parts.append(f"   Conflicts: {len(scenario.conflicts)} detected")
            
            # Synergies
            if scenario.synergies:
                report_parts.append(f"   Synergies: {len(scenario.synergies)} identified")
        
        # Recommendations
        report_parts.append(f"\nRECOMMENDATIONS:")
        best_scenario = scenarios[0]  # Highest ranked
        report_parts.append(f"• Best Overall: {best_scenario.scenario_name}")
        report_parts.append(f"• Most Cost-Effective: {min(scenarios, key=lambda x: x.total_cost / max(x.expected_impact_percent, 1)).scenario_name}")
        report_parts.append(f"• Highest Impact: {max(scenarios, key=lambda x: x.expected_impact_percent).scenario_name}")
        report_parts.append(f"• Fastest Implementation: {min(scenarios, key=lambda x: x.timeline_days).scenario_name}")
        
        return "\n".join(report_parts)

def initialize_scenario_engine() -> ScenarioComparisonEngine:
    """Initialize the scenario comparison engine"""
    try:
        engine = ScenarioComparisonEngine()
        logger.info("Scenario comparison engine initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize scenario engine: {e}")
        return None
