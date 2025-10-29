import networkx as nx
import json
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)

class InterventionDependencyGraph:
    """Graph-based system for modeling intervention dependencies, conflicts, and synergies"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.conflict_graph = nx.Graph()
        self.synergy_graph = nx.Graph()
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize dependency, conflict, and synergy rules"""
        
        # Dependency rules (prerequisites)
        self.dependency_rules = {
            # Sign dependencies
            "stop_sign": ["stop_line_marking", "advance_warning_sign"],
            "yield_sign": ["give_way_line"],
            "speed_limit_sign": ["speed_hump", "rumble_strips"],
            "school_zone_sign": ["school_zone_marking", "speed_hump"],
            "hospital_zone_sign": ["hospital_zone_marking", "no_horn_sign"],
            
            # Marking dependencies
            "zebra_crossing": ["pedestrian_warning_sign", "advance_warning_sign"],
            "stop_line": ["stop_sign"],
            "give_way_line": ["yield_sign"],
            "lane_marking": ["center_line", "edge_line"],
            
            # Infrastructure dependencies
            "traffic_signal": ["stop_line", "pedestrian_crossing"],
            "speed_hump": ["speed_hump_marking", "advance_warning_sign"],
            "guard_rail": ["street_lighting"],
            "crash_barrier": ["street_lighting"],
            "pedestrian_bridge": ["footpath", "approach_signage"],
            "underpass": ["footpath", "lighting"],
            
            # Traffic calming dependencies
            "chicane": ["advance_warning_sign", "speed_limit_sign"],
            "traffic_circle": ["yield_sign", "lane_marking"],
            "raised_crosswalk": ["pedestrian_warning_sign", "advance_warning_sign"],
            "curb_extension": ["lane_marking", "parking_restrictions"]
        }
        
        # Conflict rules (incompatible interventions)
        self.conflict_rules = {
            # Sign conflicts
            "stop_sign": ["yield_sign", "traffic_signal"],
            "yield_sign": ["stop_sign"],
            "no_entry_sign": ["one_way_sign"],
            "speed_limit_30": ["speed_limit_50", "speed_limit_60"],
            "speed_limit_50": ["speed_limit_30", "speed_limit_60"],
            
            # Infrastructure conflicts
            "speed_hump": ["ambulance_route", "bus_route"],
            "traffic_circle": ["traffic_signal"],
            "chicane": ["emergency_vehicle_access"],
            "curb_extension": ["bus_stop", "loading_zone"],
            
            # Marking conflicts
            "no_parking_zone": ["parking_bay_marking"],
            "cycle_lane_marking": ["parking_bay_marking"],
            "bus_lane_marking": ["general_lane_marking"]
        }
        
        # Synergy rules (complementary interventions)
        self.synergy_rules = {
            # Sign synergies
            "stop_sign": ["stop_line", "pedestrian_crossing_sign", "speed_limit_sign"],
            "school_zone_sign": ["school_zone_marking", "speed_hump", "pedestrian_crossing"],
            "hospital_zone_sign": ["hospital_zone_marking", "no_horn_sign", "pedestrian_crossing"],
            "railway_crossing_sign": ["stop_line", "advance_warning_sign", "barrier_gate"],
            
            # Marking synergies
            "zebra_crossing": ["pedestrian_warning_sign", "advance_warning_sign", "speed_hump"],
            "stop_line": ["stop_sign", "traffic_signal"],
            "lane_marking": ["center_line", "edge_line", "arrow_marking"],
            
            # Infrastructure synergies
            "traffic_signal": ["stop_line", "pedestrian_crossing", "advance_warning_sign"],
            "speed_hump": ["speed_hump_marking", "advance_warning_sign", "speed_limit_sign"],
            "guard_rail": ["street_lighting", "reflective_marking"],
            "pedestrian_bridge": ["approach_signage", "footpath", "lighting"],
            
            # Traffic calming synergies
            "chicane": ["speed_limit_sign", "advance_warning_sign", "lane_marking"],
            "traffic_circle": ["yield_sign", "lane_marking", "center_island"],
            "raised_crosswalk": ["pedestrian_warning_sign", "advance_warning_sign", "speed_hump"]
        }
    
    def build_graph(self, interventions: List[Dict[str, Any]]):
        """Build the dependency graph from interventions data"""
        try:
            logger.info("Building intervention dependency graph...")
            
            # Add all interventions as nodes
            for intervention in interventions:
                intervention_id = intervention['intervention_id']
                intervention_name = intervention['intervention_name'].lower()
                category = intervention['category']
                
                # Add node with metadata
                self.graph.add_node(
                    intervention_id,
                    name=intervention['intervention_name'],
                    category=category,
                    cost=intervention['cost_estimate']['total'],
                    impact=intervention['predicted_impact']['accident_reduction_percent']
                )
            
            # Add dependency edges
            self._add_dependencies(interventions)
            
            # Add conflict edges
            self._add_conflicts(interventions)
            
            # Add synergy edges
            self._add_synergies(interventions)
            
            logger.info(f"Graph built successfully with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build dependency graph: {e}")
            raise
    
    def _add_dependencies(self, interventions: List[Dict[str, Any]]):
        """Add dependency edges to the graph"""
        for intervention in interventions:
            intervention_id = intervention['intervention_id']
            intervention_name = intervention['intervention_name'].lower()
            
            # Check for dependencies based on rules
            for rule_key, dependencies in self.dependency_rules.items():
                if any(keyword in intervention_name for keyword in rule_key.split('_')):
                    for dep in dependencies:
                        # Find matching intervention
                        matching_interventions = self._find_matching_interventions(interventions, dep)
                        for match in matching_interventions:
                            self.graph.add_edge(match['intervention_id'], intervention_id, 
                                             relationship='dependency', type='prerequisite')
    
    def _add_conflicts(self, interventions: List[Dict[str, Any]]):
        """Add conflict edges to the conflict graph"""
        for intervention in interventions:
            intervention_id = intervention['intervention_id']
            intervention_name = intervention['intervention_name'].lower()
            
            # Check for conflicts based on rules
            for rule_key, conflicts in self.conflict_rules.items():
                if any(keyword in intervention_name for keyword in rule_key.split('_')):
                    for conflict in conflicts:
                        # Find matching intervention
                        matching_interventions = self._find_matching_interventions(interventions, conflict)
                        for match in matching_interventions:
                            self.conflict_graph.add_edge(intervention_id, match['intervention_id'], 
                                                      relationship='conflict')
    
    def _add_synergies(self, interventions: List[Dict[str, Any]]):
        """Add synergy edges to the synergy graph"""
        for intervention in interventions:
            intervention_id = intervention['intervention_id']
            intervention_name = intervention['intervention_name'].lower()
            
            # Check for synergies based on rules
            for rule_key, synergies in self.synergy_rules.items():
                if any(keyword in intervention_name for keyword in rule_key.split('_')):
                    for synergy in synergies:
                        # Find matching intervention
                        matching_interventions = self._find_matching_interventions(interventions, synergy)
                        for match in matching_interventions:
                            self.synergy_graph.add_edge(intervention_id, match['intervention_id'], 
                                                     relationship='synergy')
    
    def _find_matching_interventions(self, interventions: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
        """Find interventions matching a keyword"""
        matches = []
        keyword_lower = keyword.lower()
        
        for intervention in interventions:
            intervention_name = intervention['intervention_name'].lower()
            if keyword_lower in intervention_name:
                matches.append(intervention)
        
        return matches
    
    def check_dependencies(self, intervention_ids: List[str]) -> Dict[str, List[str]]:
        """Check dependencies for a set of interventions"""
        dependencies = {}
        
        for intervention_id in intervention_ids:
            if intervention_id in self.graph:
                # Get direct dependencies
                deps = list(self.graph.predecessors(intervention_id))
                dependencies[intervention_id] = deps
            else:
                dependencies[intervention_id] = []
        
        return dependencies
    
    def check_conflicts(self, intervention_ids: List[str]) -> List[Tuple[str, str]]:
        """Check conflicts between interventions"""
        conflicts = []
        
        for i, id1 in enumerate(intervention_ids):
            for id2 in intervention_ids[i+1:]:
                if self.conflict_graph.has_edge(id1, id2):
                    conflicts.append((id1, id2))
        
        return conflicts
    
    def find_synergies(self, intervention_ids: List[str]) -> List[Tuple[str, str]]:
        """Find synergies between interventions"""
        synergies = []
        
        for i, id1 in enumerate(intervention_ids):
            for id2 in intervention_ids[i+1:]:
                if self.synergy_graph.has_edge(id1, id2):
                    synergies.append((id1, id2))
        
        return synergies
    
    def get_prerequisites(self, intervention_id: str) -> List[str]:
        """Get all prerequisites for an intervention"""
        prerequisites = []
        
        if intervention_id in self.graph:
            # Get direct prerequisites
            direct_prereqs = list(self.graph.predecessors(intervention_id))
            prerequisites.extend(direct_prereqs)
            
            # Get transitive prerequisites
            for prereq in direct_prereqs:
                transitive_prereqs = self.get_prerequisites(prereq)
                prerequisites.extend(transitive_prereqs)
        
        return list(set(prerequisites))  # Remove duplicates
    
    def validate_intervention_set(self, intervention_ids: List[str]) -> Dict[str, Any]:
        """Validate a set of interventions for conflicts and missing dependencies"""
        validation_result = {
            'valid': True,
            'conflicts': [],
            'missing_dependencies': [],
            'synergies': [],
            'warnings': []
        }
        
        # Check for conflicts
        conflicts = self.check_conflicts(intervention_ids)
        if conflicts:
            validation_result['conflicts'] = conflicts
            validation_result['valid'] = False
        
        # Check for missing dependencies
        all_prereqs = set()
        for intervention_id in intervention_ids:
            prereqs = self.get_prerequisites(intervention_id)
            missing_prereqs = [p for p in prereqs if p not in intervention_ids]
            if missing_prereqs:
                validation_result['missing_dependencies'].append({
                    'intervention': intervention_id,
                    'missing_prerequisites': missing_prereqs
                })
                validation_result['valid'] = False
        
        # Find synergies
        synergies = self.find_synergies(intervention_ids)
        validation_result['synergies'] = synergies
        
        # Generate warnings
        if len(intervention_ids) > 5:
            validation_result['warnings'].append("Large number of interventions may increase implementation complexity")
        
        return validation_result
    
    def optimize_intervention_set(self, intervention_ids: List[str], max_cost: int = None) -> List[str]:
        """Optimize intervention set by removing conflicts and adding synergies"""
        optimized_set = intervention_ids.copy()
        
        # Remove conflicts
        conflicts = self.check_conflicts(optimized_set)
        for id1, id2 in conflicts:
            # Keep the intervention with higher impact
            impact1 = self.graph.nodes[id1]['impact']
            impact2 = self.graph.nodes[id2]['impact']
            
            if impact1 > impact2:
                optimized_set.remove(id2)
            else:
                optimized_set.remove(id1)
        
        # Add synergies if cost allows
        if max_cost:
            current_cost = sum(self.graph.nodes[i]['cost'] for i in optimized_set)
            
            for intervention_id in optimized_set:
                synergies = list(self.synergy_graph.neighbors(intervention_id))
                for synergy_id in synergies:
                    if synergy_id not in optimized_set:
                        synergy_cost = self.graph.nodes[synergy_id]['cost']
                        if current_cost + synergy_cost <= max_cost:
                            optimized_set.append(synergy_id)
                            current_cost += synergy_cost
        
        return optimized_set
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dependency graph"""
        return {
            'total_interventions': self.graph.number_of_nodes(),
            'total_dependencies': self.graph.number_of_edges(),
            'total_conflicts': self.conflict_graph.number_of_edges(),
            'total_synergies': self.synergy_graph.number_of_edges(),
            'average_dependencies_per_intervention': self.graph.number_of_edges() / max(1, self.graph.number_of_nodes()),
            'most_connected_intervention': max(self.graph.degree(), key=lambda x: x[1])[0] if self.graph.number_of_nodes() > 0 else None
        }
    
    def export_graph(self, output_file: str):
        """Export graph to file for visualization"""
        try:
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'name': data['name'],
                        'category': data['category'],
                        'cost': data['cost'],
                        'impact': data['impact']
                    }
                    for node, data in self.graph.nodes(data=True)
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': data.get('relationship', 'dependency'),
                        'type': data.get('type', 'prerequisite')
                    }
                    for edge, data in self.graph.edges(data=True)
                ],
                'conflicts': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': 'conflict'
                    }
                    for edge in self.conflict_graph.edges()
                ],
                'synergies': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': 'synergy'
                    }
                    for edge in self.synergy_graph.edges()
                ]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Graph exported to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            raise

def initialize_dependency_graph(interventions_file: str = "data/interventions/interventions.json") -> InterventionDependencyGraph:
    """Initialize dependency graph system"""
    try:
        # Load interventions data
        with open(interventions_file, 'r', encoding='utf-8') as f:
            interventions = json.load(f)
        
        # Create dependency graph
        graph = InterventionDependencyGraph()
        graph.build_graph(interventions)
        
        return graph
        
    except Exception as e:
        logger.error(f"Failed to initialize dependency graph: {e}")
        raise
