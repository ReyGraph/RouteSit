#!/usr/bin/env python3
"""
Dependency Graph System
NetworkX-based intervention dependency and conflict detection using IRC/MoRTH rules
"""

import json
import logging
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """Types of dependencies between interventions"""
    PREREQUISITE = "prerequisite"  # Must be done before
    CONFLICT = "conflict"          # Cannot be done together
    SYNERGY = "synergy"           # Work well together
    SEQUENCE = "sequence"         # Should be done in order

@dataclass
class DependencyRule:
    """Dependency rule between interventions"""
    source_intervention: str
    target_intervention: str
    dependency_type: DependencyType
    description: str
    irc_reference: str
    confidence: float = 1.0

@dataclass
class ConflictAnalysis:
    """Analysis of intervention conflicts and dependencies"""
    conflicts: List[Tuple[str, str, str]]  # (intervention1, intervention2, reason)
    prerequisites: List[Tuple[str, str, str]]  # (intervention, prerequisite, reason)
    synergies: List[Tuple[str, str, str]]  # (intervention1, intervention2, benefit)
    optimal_sequence: List[str]
    total_additional_cost: int
    timeline_impact_days: int

class DependencyGraph:
    """NetworkX-based dependency graph for road safety interventions"""
    
    def __init__(self, rules_path: str = "config/dependency_rules.json"):
        self.rules_path = Path(rules_path)
        self.graph = nx.DiGraph()
        self.conflict_graph = nx.Graph()
        self.synergy_graph = nx.Graph()
        self.rules = []
        
        self._load_dependency_rules()
        self._build_graphs()
    
    def _load_dependency_rules(self):
        """Load dependency rules from IRC/MoRTH standards"""
        if self.rules_path.exists():
            try:
                with open(self.rules_path, 'r') as f:
                    data = json.load(f)
                    self.rules = [DependencyRule(**rule) for rule in data.get("rules", [])]
                logger.info(f"Loaded {len(self.rules)} dependency rules")
            except Exception as e:
                logger.error(f"Failed to load dependency rules: {e}")
                self._create_default_rules()
        else:
            logger.warning("Dependency rules file not found, creating default rules")
            self._create_default_rules()
    
    def _create_default_rules(self):
        """Create default dependency rules based on IRC/MoRTH standards"""
        self.rules = [
            # Prerequisites
            DependencyRule(
                source_intervention="Install Pedestrian Crossing",
                target_intervention="Install Warning Sign",
                dependency_type=DependencyType.PREREQUISITE,
                description="Warning signs must be installed 50m before pedestrian crossing",
                irc_reference="IRC 67-2022 Clause 14.4",
                confidence=1.0
            ),
            DependencyRule(
                source_intervention="Install Speed Hump",
                target_intervention="Install Warning Sign",
                dependency_type=DependencyType.PREREQUISITE,
                description="Speed hump warning signs required 30m before hump",
                irc_reference="IRC 103-2012 Clause 8.2",
                confidence=1.0
            ),
            DependencyRule(
                source_intervention="Install Traffic Signal",
                target_intervention="Install Stop Line",
                dependency_type=DependencyType.PREREQUISITE,
                description="Stop line marking required for traffic signals",
                irc_reference="IRC 35-2015 Clause 7.1",
                confidence=1.0
            ),
            DependencyRule(
                source_intervention="Install Street Lighting",
                target_intervention="Install Warning Sign",
                dependency_type=DependencyType.PREREQUISITE,
                description="Warning signs required for lighting installation zones",
                irc_reference="MoRTH Guidelines 2018",
                confidence=0.8
            ),
            
            # Conflicts
            DependencyRule(
                source_intervention="Install Speed Hump",
                target_intervention="Install Speed Camera",
                dependency_type=DependencyType.CONFLICT,
                description="Speed humps and cameras serve similar purposes, choose one",
                irc_reference="IRC 103-2012 Clause 8.5",
                confidence=0.9
            ),
            DependencyRule(
                source_intervention="Install Speed Hump",
                target_intervention="Emergency Vehicle Route",
                dependency_type=DependencyType.CONFLICT,
                description="Speed humps impede emergency vehicles",
                irc_reference="IRC 103-2012 Clause 8.6",
                confidence=1.0
            ),
            DependencyRule(
                source_intervention="Install Rumble Strip",
                target_intervention="Install Speed Hump",
                dependency_type=DependencyType.CONFLICT,
                description="Rumble strips and speed humps are redundant",
                irc_reference="IRC 103-2012 Clause 8.7",
                confidence=0.8
            ),
            
            # Synergies
            DependencyRule(
                source_intervention="Repaint Road Marking",
                target_intervention="Install Street Lighting",
                dependency_type=DependencyType.SYNERGY,
                description="Lighting improves marking visibility at night",
                irc_reference="IRC 35-2015 Clause 7.3",
                confidence=0.9
            ),
            DependencyRule(
                source_intervention="Install Warning Sign",
                target_intervention="Install Speed Camera",
                dependency_type=DependencyType.SYNERGY,
                description="Warning signs increase camera effectiveness",
                irc_reference="MoRTH Guidelines 2018",
                confidence=0.8
            ),
            DependencyRule(
                source_intervention="Install Pedestrian Crossing",
                target_intervention="Install Guard Rail",
                dependency_type=DependencyType.SYNERGY,
                description="Guard rails channel pedestrians to crossing",
                irc_reference="IRC 67-2022 Clause 14.5",
                confidence=0.9
            ),
            
            # Sequences
            DependencyRule(
                source_intervention="Install Road Sign",
                target_intervention="Install Speed Camera",
                dependency_type=DependencyType.SEQUENCE,
                description="Signs should be installed before cameras for compliance",
                irc_reference="MoRTH Guidelines 2018",
                confidence=0.8
            ),
            DependencyRule(
                source_intervention="Install Stop Line",
                target_intervention="Install Traffic Signal",
                dependency_type=DependencyType.SEQUENCE,
                description="Stop line must be painted before signal activation",
                irc_reference="IRC 35-2015 Clause 7.1",
                confidence=1.0
            )
        ]
        
        # Save default rules
        self._save_rules()
    
    def _save_rules(self):
        """Save rules to file"""
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "rules": [
                    {
                        "source_intervention": rule.source_intervention,
                        "target_intervention": rule.target_intervention,
                        "dependency_type": rule.dependency_type.value,
                        "description": rule.description,
                        "irc_reference": rule.irc_reference,
                        "confidence": rule.confidence
                    }
                    for rule in self.rules
                ]
            }
            
            with open(self.rules_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save dependency rules: {e}")
    
    def _build_graphs(self):
        """Build NetworkX graphs from dependency rules"""
        # Clear existing graphs
        self.graph.clear()
        self.conflict_graph.clear()
        self.synergy_graph.clear()
        
        # Add nodes for all interventions mentioned in rules
        all_interventions = set()
        for rule in self.rules:
            all_interventions.add(rule.source_intervention)
            all_interventions.add(rule.target_intervention)
        
        for intervention in all_interventions:
            self.graph.add_node(intervention)
            self.conflict_graph.add_node(intervention)
            self.synergy_graph.add_node(intervention)
        
        # Add edges based on dependency types
        for rule in self.rules:
            if rule.dependency_type == DependencyType.PREREQUISITE:
                self.graph.add_edge(
                    rule.target_intervention, 
                    rule.source_intervention,
                    description=rule.description,
                    reference=rule.irc_reference,
                    confidence=rule.confidence
                )
            elif rule.dependency_type == DependencyType.CONFLICT:
                self.conflict_graph.add_edge(
                    rule.source_intervention,
                    rule.target_intervention,
                    description=rule.description,
                    reference=rule.irc_reference,
                    confidence=rule.confidence
                )
            elif rule.dependency_type == DependencyType.SYNERGY:
                self.synergy_graph.add_edge(
                    rule.source_intervention,
                    rule.target_intervention,
                    description=rule.description,
                    reference=rule.irc_reference,
                    confidence=rule.confidence
                )
            elif rule.dependency_type == DependencyType.SEQUENCE:
                self.graph.add_edge(
                    rule.source_intervention,
                    rule.target_intervention,
                    description=rule.description,
                    reference=rule.irc_reference,
                    confidence=rule.confidence
                )
        
        logger.info(f"Built dependency graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        logger.info(f"Built conflict graph with {self.conflict_graph.number_of_nodes()} nodes and {self.conflict_graph.number_of_edges()} edges")
        logger.info(f"Built synergy graph with {self.synergy_graph.number_of_nodes()} nodes and {self.synergy_graph.number_of_edges()} edges")
    
    def analyze_interventions(self, interventions: List[str]) -> ConflictAnalysis:
        """Analyze conflicts and dependencies for a set of interventions"""
        conflicts = []
        prerequisites = []
        synergies = []
        
        # Check for conflicts
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                if self.conflict_graph.has_edge(intervention1, intervention2):
                    edge_data = self.conflict_graph[intervention1][intervention2]
                    conflicts.append((
                        intervention1,
                        intervention2,
                        edge_data.get('description', 'Conflict detected')
                    ))
        
        # Find prerequisites
        for intervention in interventions:
            predecessors = list(self.graph.predecessors(intervention))
            for pred in predecessors:
                if pred not in interventions:
                    edge_data = self.graph[pred][intervention]
                    prerequisites.append((
                        intervention,
                        pred,
                        edge_data.get('description', 'Prerequisite required')
                    ))
        
        # Find synergies
        for i, intervention1 in enumerate(interventions):
            for intervention2 in interventions[i+1:]:
                if self.synergy_graph.has_edge(intervention1, intervention2):
                    edge_data = self.synergy_graph[intervention1][intervention2]
                    synergies.append((
                        intervention1,
                        intervention2,
                        edge_data.get('description', 'Synergistic effect')
                    ))
        
        # Calculate optimal sequence
        optimal_sequence = self._calculate_optimal_sequence(interventions)
        
        # Estimate additional costs and timeline impact
        additional_cost = len(prerequisites) * 15000  # Rs 15k per prerequisite
        timeline_impact = len(prerequisites) * 2  # 2 days per prerequisite
        
        return ConflictAnalysis(
            conflicts=conflicts,
            prerequisites=prerequisites,
            synergies=synergies,
            optimal_sequence=optimal_sequence,
            total_additional_cost=additional_cost,
            timeline_impact_days=timeline_impact
        )
    
    def _calculate_optimal_sequence(self, interventions: List[str]) -> List[str]:
        """Calculate optimal implementation sequence using topological sort"""
        try:
            # Create subgraph with only the interventions we're analyzing
            subgraph = self.graph.subgraph(interventions)
            
            # Try topological sort
            if nx.is_directed_acyclic_graph(subgraph):
                sequence = list(nx.topological_sort(subgraph))
                return sequence
            else:
                # If there are cycles, use a different approach
                logger.warning("Cycle detected in dependency graph, using alternative sequencing")
                return self._alternative_sequencing(interventions)
                
        except Exception as e:
            logger.error(f"Failed to calculate optimal sequence: {e}")
            return interventions  # Return original order as fallback
    
    def _alternative_sequencing(self, interventions: List[str]) -> List[str]:
        """Alternative sequencing when cycles are detected"""
        # Simple heuristic: prioritize interventions with fewer dependencies
        intervention_deps = {}
        
        for intervention in interventions:
            deps = list(self.graph.predecessors(intervention))
            intervention_deps[intervention] = len(deps)
        
        # Sort by number of dependencies (ascending)
        sorted_interventions = sorted(interventions, key=lambda x: intervention_deps.get(x, 0))
        
        return sorted_interventions
    
    def get_intervention_dependencies(self, intervention: str) -> Dict[str, List[str]]:
        """Get all dependencies for a specific intervention"""
        if intervention not in self.graph:
            return {"prerequisites": [], "conflicts": [], "synergies": []}
        
        prerequisites = list(self.graph.predecessors(intervention))
        successors = list(self.graph.successors(intervention))
        conflicts = list(self.conflict_graph.neighbors(intervention))
        synergies = list(self.synergy_graph.neighbors(intervention))
        
        return {
            "prerequisites": prerequisites,
            "successors": successors,
            "conflicts": conflicts,
            "synergies": synergies
        }
    
    def validate_intervention_plan(self, interventions: List[str]) -> Dict[str, Any]:
        """Validate a complete intervention plan"""
        analysis = self.analyze_interventions(interventions)
        
        validation_result = {
            "is_valid": len(analysis.conflicts) == 0,
            "conflicts": analysis.conflicts,
            "missing_prerequisites": analysis.prerequisites,
            "synergies": analysis.synergies,
            "recommended_sequence": analysis.optimal_sequence,
            "additional_cost": analysis.total_additional_cost,
            "timeline_impact": analysis.timeline_impact_days,
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis.conflicts:
            validation_result["recommendations"].append(
                f"Remove conflicting interventions: {', '.join([f'{c[0]} vs {c[1]}' for c in analysis.conflicts])}"
            )
        
        if analysis.prerequisites:
            validation_result["recommendations"].append(
                f"Add missing prerequisites: {', '.join([f'{p[1]} for {p[0]}' for p in analysis.prerequisites])}"
            )
        
        if analysis.synergies:
            validation_result["recommendations"].append(
                f"Consider synergistic combinations: {', '.join([f'{s[0]} + {s[1]}' for s in analysis.synergies])}"
            )
        
        return validation_result
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dependency graphs"""
        return {
            "dependency_graph": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "is_dag": nx.is_directed_acyclic_graph(self.graph)
            },
            "conflict_graph": {
                "nodes": self.conflict_graph.number_of_nodes(),
                "edges": self.conflict_graph.number_of_edges(),
                "connected_components": nx.number_connected_components(self.conflict_graph)
            },
            "synergy_graph": {
                "nodes": self.synergy_graph.number_of_nodes(),
                "edges": self.synergy_graph.number_of_edges(),
                "connected_components": nx.number_connected_components(self.synergy_graph)
            },
            "total_rules": len(self.rules)
        }

def main():
    """Test the dependency graph system"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Dependency Graph System...")
    
    graph = DependencyGraph()
    
    # Test intervention analysis
    test_interventions = [
        "Install Pedestrian Crossing",
        "Install Speed Hump", 
        "Install Warning Sign",
        "Install Speed Camera"
    ]
    
    analysis = graph.analyze_interventions(test_interventions)
    
    print(f"Conflicts: {len(analysis.conflicts)}")
    for conflict in analysis.conflicts:
        print(f"  - {conflict[0]} vs {conflict[1]}: {conflict[2]}")
    
    print(f"Prerequisites: {len(analysis.prerequisites)}")
    for prereq in analysis.prerequisites:
        print(f"  - {prereq[0]} needs {prereq[1]}: {prereq[2]}")
    
    print(f"Synergies: {len(analysis.synergies)}")
    for synergy in analysis.synergies:
        print(f"  - {synergy[0]} + {synergy[1]}: {synergy[2]}")
    
    print(f"Optimal sequence: {analysis.optimal_sequence}")
    print(f"Additional cost: Rs {analysis.total_additional_cost:,}")
    print(f"Timeline impact: {analysis.timeline_impact_days} days")
    
    # Test validation
    validation = graph.validate_intervention_plan(test_interventions)
    print(f"\nPlan valid: {validation['is_valid']}")
    print(f"Recommendations: {validation['recommendations']}")
    
    # Test single intervention dependencies
    deps = graph.get_intervention_dependencies("Install Pedestrian Crossing")
    print(f"\nDependencies for Pedestrian Crossing:")
    print(f"  Prerequisites: {deps['prerequisites']}")
    print(f"  Conflicts: {deps['conflicts']}")
    print(f"  Synergies: {deps['synergies']}")

if __name__ == "__main__":
    main()
