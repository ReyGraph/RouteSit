import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .vector_search import InterventionRetriever, initialize_vector_search
from .llm_engine import LocalLLMEngine
from ..data.dependency_graph import InterventionDependencyGraph, initialize_dependency_graph
from .optimization import CostBenefitOptimizer, OptimizationResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RoutesitReasoningEngine:
    """Main reasoning engine that orchestrates all AI components"""
    
    def __init__(self):
        self.retriever = None
        self.llm_engine = None
        self.dependency_graph = None
        self.optimizer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        try:
            logger.info("Initializing Routesit AI reasoning engine...")
            
            # Initialize vector search
            self.retriever = initialize_vector_search()
            
            # Initialize LLM engine
            self.llm_engine = LocalLLMEngine()
            
            # Initialize dependency graph
            self.dependency_graph = initialize_dependency_graph()
            
            # Initialize optimizer
            self.optimizer = CostBenefitOptimizer()
            
            logger.info("Routesit AI reasoning engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            raise
    
    def process_query(self, 
                     query: str, 
                     context: Optional[Dict[str, Any]] = None,
                     budget_constraint: Optional[float] = None,
                     timeline_constraint: Optional[int] = None,
                     min_impact_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a road safety query and generate comprehensive recommendations
        
        Args:
            query: Problem description
            context: Additional context (road type, traffic, etc.)
            budget_constraint: Maximum budget
            timeline_constraint: Maximum timeline
            min_impact_threshold: Minimum impact requirement
            
        Returns:
            Comprehensive analysis and recommendations
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Retrieve relevant interventions
            interventions = self._retrieve_interventions(query, context)
            
            if not interventions:
                return self._create_empty_response(query, "No relevant interventions found")
            
            # Step 2: Validate dependencies and conflicts
            validation_result = self._validate_interventions(interventions)
            
            # Step 3: Optimize intervention selection
            optimization_results = self._optimize_interventions(
                interventions, budget_constraint, timeline_constraint, min_impact_threshold
            )
            
            # Step 4: Generate reasoning and explanations
            reasoning = self._generate_reasoning(query, interventions, context)
            explanations = self._generate_explanations(query, interventions)
            
            # Step 5: Create comprehensive response
            response = self._create_comprehensive_response(
                query, interventions, optimization_results, validation_result, 
                reasoning, explanations, context
            )
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return self._create_error_response(query, str(e))
    
    def _retrieve_interventions(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant interventions using vector search"""
        try:
            # Basic retrieval
            interventions = self.retriever.retrieve_interventions(query, n_results=20)
            
            # Apply context-based filtering
            if context:
                interventions = self._apply_context_filtering(interventions, context)
            
            # Sort by relevance and impact
            interventions.sort(key=lambda x: x.get('relevance_score', 0) * x['predicted_impact']['accident_reduction_percent'], reverse=True)
            
            return interventions[:15]  # Limit to top 15
            
        except Exception as e:
            logger.error(f"Intervention retrieval failed: {e}")
            return []
    
    def _apply_context_filtering(self, interventions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based filtering to interventions"""
        filtered_interventions = []
        
        for intervention in interventions:
            # Filter by road type
            road_type = context.get('road_type', '').lower()
            if road_type:
                if road_type in ['highway', 'expressway'] and intervention['category'] == 'traffic_calming':
                    continue  # Skip traffic calming on highways
                elif road_type in ['residential', 'school_zone'] and intervention['category'] == 'infrastructure':
                    # Prioritize traffic calming for residential areas
                    intervention['relevance_score'] = intervention.get('relevance_score', 0) * 1.2
            
            # Filter by traffic volume
            traffic_volume = context.get('traffic_volume', '').lower()
            if traffic_volume == 'high' and intervention['implementation_timeline'] > 7:
                continue  # Skip long-term interventions for high traffic areas
            
            # Filter by speed limit
            speed_limit = context.get('speed_limit', 0)
            if speed_limit > 60 and intervention['category'] == 'traffic_calming':
                continue  # Skip traffic calming on high-speed roads
            
            filtered_interventions.append(intervention)
        
        return filtered_interventions
    
    def _validate_interventions(self, interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate interventions for dependencies and conflicts"""
        try:
            intervention_ids = [intv['intervention_id'] for intv in interventions]
            validation_result = self.dependency_graph.validate_intervention_set(intervention_ids)
            
            # Add intervention names to validation result
            if validation_result['conflicts']:
                for i, (id1, id2) in enumerate(validation_result['conflicts']):
                    name1 = next((intv['intervention_name'] for intv in interventions if intv['intervention_id'] == id1), id1)
                    name2 = next((intv['intervention_name'] for intv in interventions if intv['intervention_id'] == id2), id2)
                    validation_result['conflicts'][i] = (name1, name2)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {'valid': True, 'conflicts': [], 'missing_dependencies': [], 'synergies': [], 'warnings': []}
    
    def _optimize_interventions(self, 
                               interventions: List[Dict[str, Any]], 
                               budget_constraint: Optional[float] = None,
                               timeline_constraint: Optional[int] = None,
                               min_impact_threshold: Optional[float] = None) -> List[OptimizationResult]:
        """Optimize intervention selection"""
        try:
            return self.optimizer.optimize_interventions(
                interventions, budget_constraint, timeline_constraint, min_impact_threshold
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return []
    
    def _generate_reasoning(self, query: str, interventions: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        """Generate reasoning for recommendations"""
        try:
            return self.llm_engine.generate_reasoning(query, interventions, context)
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"Analysis of the problem '{query}' suggests implementing {len(interventions)} key interventions to improve road safety."
    
    def _generate_explanations(self, query: str, interventions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate explanations for each intervention"""
        explanations = {}
        
        try:
            for intervention in interventions[:5]:  # Limit to top 5
                explanation = self.llm_engine.generate_explanation(intervention, query)
                explanations[intervention['intervention_id']] = explanation
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
        
        return explanations
    
    def _create_comprehensive_response(self, 
                                      query: str, 
                                      interventions: List[Dict[str, Any]], 
                                      optimization_results: List[OptimizationResult],
                                      validation_result: Dict[str, Any],
                                      reasoning: str,
                                      explanations: Dict[str, str],
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive response"""
        
        response = {
            'query': query,
            'context': context,
            'timestamp': self._get_timestamp(),
            'analysis': {
                'total_interventions_found': len(interventions),
                'validation_result': validation_result,
                'reasoning': reasoning
            },
            'recommendations': {
                'optimization_scenarios': [],
                'best_scenario': None,
                'scenario_comparison': {}
            },
            'interventions': {
                'detailed_list': interventions[:10],  # Top 10 interventions
                'explanations': explanations
            },
            'implementation': {
                'estimated_timeline': 0,
                'estimated_cost': 0,
                'expected_impact': 0
            },
            'next_steps': []
        }
        
        # Add optimization scenarios
        if optimization_results:
            response['recommendations']['optimization_scenarios'] = [
                {
                    'strategy': result.strategy,
                    'interventions': result.selected_interventions,
                    'total_cost': result.total_cost,
                    'total_impact': result.total_impact,
                    'cost_effectiveness': result.cost_effectiveness,
                    'timeline': result.implementation_timeline,
                    'confidence': result.confidence_score
                }
                for result in optimization_results
            ]
            
            # Set best scenario
            best_scenario = optimization_results[0]
            response['recommendations']['best_scenario'] = {
                'strategy': best_scenario.strategy,
                'interventions': best_scenario.selected_interventions,
                'total_cost': best_scenario.total_cost,
                'total_impact': best_scenario.total_impact,
                'cost_effectiveness': best_scenario.cost_effectiveness,
                'timeline': best_scenario.implementation_timeline,
                'confidence': best_scenario.confidence_score
            }
            
            # Add scenario comparison
            response['recommendations']['scenario_comparison'] = self.optimizer.compare_scenarios(optimization_results)
            
            # Set implementation details
            response['implementation'] = {
                'estimated_timeline': best_scenario.implementation_timeline,
                'estimated_cost': best_scenario.total_cost,
                'expected_impact': best_scenario.total_impact
            }
        
        # Add next steps
        response['next_steps'] = self._generate_next_steps(validation_result, optimization_results)
        
        return response
    
    def _generate_next_steps(self, validation_result: Dict[str, Any], optimization_results: List[OptimizationResult]) -> List[str]:
        """Generate next steps for implementation"""
        next_steps = []
        
        if optimization_results:
            best_scenario = optimization_results[0]
            next_steps.append(f"Implement {best_scenario.strategy} strategy with {len(best_scenario.selected_interventions)} interventions")
            next_steps.append(f"Budget allocation: ₹{best_scenario.total_cost:,.0f}")
            next_steps.append(f"Timeline: {best_scenario.implementation_timeline} days")
        
        if validation_result.get('conflicts'):
            next_steps.append("Resolve intervention conflicts before implementation")
        
        if validation_result.get('missing_dependencies'):
            next_steps.append("Address missing prerequisites")
        
        next_steps.append("Obtain local authority approvals")
        next_steps.append("Conduct site survey and traffic analysis")
        next_steps.append("Establish maintenance schedule")
        
        return next_steps
    
    def _create_empty_response(self, query: str, message: str) -> Dict[str, Any]:
        """Create response when no interventions are found"""
        return {
            'query': query,
            'timestamp': self._get_timestamp(),
            'analysis': {
                'total_interventions_found': 0,
                'message': message
            },
            'recommendations': {
                'optimization_scenarios': [],
                'best_scenario': None
            },
            'interventions': {
                'detailed_list': [],
                'explanations': {}
            },
            'implementation': {
                'estimated_timeline': 0,
                'estimated_cost': 0,
                'expected_impact': 0
            },
            'next_steps': ['Consider expanding the intervention database', 'Refine the problem description']
        }
    
    def _create_error_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'query': query,
            'timestamp': self._get_timestamp(),
            'error': error_message,
            'analysis': {
                'total_interventions_found': 0,
                'message': 'Processing failed'
            },
            'recommendations': {
                'optimization_scenarios': [],
                'best_scenario': None
            },
            'interventions': {
                'detailed_list': [],
                'explanations': {}
            },
            'implementation': {
                'estimated_timeline': 0,
                'estimated_cost': 0,
                'expected_impact': 0
            },
            'next_steps': ['Please try again with a different query', 'Contact support if the issue persists']
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            status = {
                'system_status': 'operational',
                'components': {
                    'vector_search': 'operational' if self.retriever else 'failed',
                    'llm_engine': 'operational' if self.llm_engine else 'failed',
                    'dependency_graph': 'operational' if self.dependency_graph else 'failed',
                    'optimizer': 'operational' if self.optimizer else 'failed'
                },
                'statistics': {}
            }
            
            # Add statistics
            if self.retriever:
                status['statistics']['vector_search'] = self.retriever.vector_engine.get_collection_stats()
            
            if self.dependency_graph:
                status['statistics']['dependency_graph'] = self.dependency_graph.get_graph_statistics()
            
            if self.llm_engine:
                status['statistics']['llm_engine'] = self.llm_engine.get_model_info()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'components': {},
                'statistics': {}
            }

def initialize_routesit_ai() -> RoutesitReasoningEngine:
    """Initialize the complete Routesit AI system"""
    try:
        logger.info("Initializing Routesit AI system...")
        engine = RoutesitReasoningEngine()
        logger.info("Routesit AI system initialized successfully!")
        return engine
    except Exception as e:
        logger.error(f"Failed to initialize Routesit AI: {e}")
        raise
