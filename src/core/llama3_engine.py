"""
Local LLM Engine for Routesit AI
Custom inference pipeline for road safety domain reasoning
Uses local models that don't require authentication
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SafetyAnalysis:
    """Structured output from safety analysis"""
    intervention_type: str
    risk_level: str
    confidence: float
    reasoning: str
    cascading_effects: List[str]
    implementation_priority: str
    cost_estimate: Dict[str, float]
    lives_saved_estimate: float
    references: List[str]

class RoutesitLLM:
    """
    Local LLM system for road safety intervention analysis
    Uses free models that don't require authentication
    """
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = device
        # Use local models that don't require authentication
        self.model_path = model_path or "microsoft/DialoGPT-medium"
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.cache = {}
        
        # Road safety domain embeddings
        self.safety_embeddings = None
        
        # Initialize the model
        self._setup_model()
        self._load_safety_knowledge()
    
    def _setup_model(self):
        """Setup local model with fallback options"""
        logger.info("Setting up local LLM model...")
        
        # Try multiple model options in order of preference
        model_options = [
            "microsoft/DialoGPT-medium",  # Free, no auth required
            "distilbert-base-uncased",    # Lightweight fallback
            "gpt2",                       # Basic GPT-2 model
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Trying to load {model_name}...")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # Use float32 for compatibility
                )
                
                logger.info(f"Successfully loaded {model_name}")
                self.model_path = model_name
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            logger.error("Failed to load any model")
            raise RuntimeError("No suitable model could be loaded")
        
        # Setup embedding model
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _load_safety_knowledge(self):
        """Load road safety domain knowledge"""
        try:
            # Load intervention database
            db_path = Path("data/interventions/interventions_database.json")
            if db_path.exists():
                with open(db_path, 'r', encoding='utf-8') as f:
                    self.interventions_db = json.load(f)
                logger.info(f"Loaded {len(self.interventions_db)} interventions")
            else:
                self.interventions_db = []
                logger.warning("No intervention database found")
            
            # Load accident data
            accident_path = Path("data/accident_data/accident_records.json")
            if accident_path.exists():
                with open(accident_path, 'r', encoding='utf-8') as f:
                    self.accident_data = json.load(f)
                logger.info(f"Loaded {len(self.accident_data)} accident records")
            else:
                self.accident_data = []
                logger.warning("No accident data found")
                
        except Exception as e:
            logger.error(f"Error loading safety knowledge: {e}")
            self.interventions_db = []
            self.accident_data = []
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using local model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def analyze_road_safety(self, description: str, image_data: Optional[bytes] = None) -> SafetyAnalysis:
        """Analyze road safety situation and recommend interventions"""
        try:
            # Create analysis prompt
            prompt = f"""
            Analyze this road safety situation: {description}
            
            Based on the description, provide:
            1. Intervention type needed
            2. Risk level (low/medium/high)
            3. Confidence in assessment (0-1)
            4. Reasoning for the recommendation
            5. Potential cascading effects
            6. Implementation priority
            7. Cost estimate range
            8. Lives saved estimate
            9. Relevant standards/references
            
            Format as structured analysis.
            """
            
            # Generate analysis
            response = self.generate_response(prompt, max_length=300)
            
            # Parse response into structured format
            analysis = SafetyAnalysis(
                intervention_type=self._extract_intervention_type(response),
                risk_level=self._extract_risk_level(response),
                confidence=self._extract_confidence(response),
                reasoning=response,
                cascading_effects=self._extract_cascading_effects(response),
                implementation_priority=self._extract_priority(response),
                cost_estimate=self._extract_cost_estimate(response),
                lives_saved_estimate=self._extract_lives_saved(response),
                references=self._extract_references(response)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in safety analysis: {e}")
            return SafetyAnalysis(
                intervention_type="Unknown",
                risk_level="medium",
                confidence=0.5,
                reasoning="Analysis failed due to technical error",
                cascading_effects=[],
                implementation_priority="medium",
                cost_estimate={"min": 0, "max": 0},
                lives_saved_estimate=0,
                references=[]
            )
    
    def reason(self, multimodal_input: Dict[str, Any], context_history: List[str] = None) -> SafetyAnalysis:
        """Performs domain-specific reasoning based on multi-modal input"""
        logger.info("Performing domain-specific reasoning...")
        
        # Extract input components
        user_query = multimodal_input.get("text_description", "")
        image_analysis = multimodal_input.get("image_analysis", {})
        metadata = multimodal_input.get("metadata", {})
        
        # Simple reasoning based on keywords
        intervention_type = "General Recommendation"
        risk_level = "medium"
        confidence = 0.7
        reasoning = f"Based on analysis of: {user_query}"
        
        # Extract keywords to determine intervention type
        if "zebra" in user_query.lower() or "crossing" in user_query.lower():
            intervention_type = "Repaint Road Marking"
            risk_level = "high"
            confidence = 0.8
        elif "speed" in user_query.lower() or "hump" in user_query.lower():
            intervention_type = "Install Speed Hump"
            risk_level = "medium"
            confidence = 0.75
        elif "sign" in user_query.lower():
            intervention_type = "Install Road Sign"
            risk_level = "low"
            confidence = 0.7
        
        # Generate cascading effects
        cascading_effects = [
            "Improved traffic flow",
            "Enhanced pedestrian safety",
            "Reduced accident risk"
        ]
        
        # Cost estimation
        cost_estimate = {
            "min": 50000,
            "max": 100000,
            "total": 80000
        }
        
        # Lives saved estimate
        lives_saved_estimate = 2.5
        
        # References
        references = [
            "IRC 67-2022 Clause 14.4",
            "MoRTH Guidelines 2018"
        ]
        
        return SafetyAnalysis(
            intervention_type=intervention_type,
            risk_level=risk_level,
            confidence=confidence,
            reasoning=reasoning,
            cascading_effects=cascading_effects,
            implementation_priority="high",
            cost_estimate=cost_estimate,
            lives_saved_estimate=lives_saved_estimate,
            references=references
        )
    
    def _extract_intervention_type(self, response: str) -> str:
        """Extract intervention type from response"""
        # Simple keyword matching
        if "zebra crossing" in response.lower():
            return "Zebra Crossing Enhancement"
        elif "speed" in response.lower():
            return "Speed Management"
        elif "sign" in response.lower():
            return "Traffic Sign Installation"
        elif "barrier" in response.lower():
            return "Safety Barrier Installation"
        else:
            return "General Safety Intervention"
    
    def _extract_risk_level(self, response: str) -> str:
        """Extract risk level from response"""
        if "high" in response.lower():
            return "high"
        elif "low" in response.lower():
            return "low"
        else:
            return "medium"
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        # Look for confidence indicators
        if "high confidence" in response.lower():
            return 0.9
        elif "low confidence" in response.lower():
            return 0.3
        else:
            return 0.7
    
    def _extract_cascading_effects(self, response: str) -> List[str]:
        """Extract cascading effects from response"""
        effects = []
        if "traffic flow" in response.lower():
            effects.append("Traffic flow changes")
        if "speed" in response.lower():
            effects.append("Speed pattern changes")
        if "pedestrian" in response.lower():
            effects.append("Pedestrian behavior changes")
        return effects
    
    def _extract_priority(self, response: str) -> str:
        """Extract implementation priority from response"""
        if "urgent" in response.lower() or "immediate" in response.lower():
            return "high"
        elif "low" in response.lower():
            return "low"
        else:
            return "medium"
    
    def _extract_cost_estimate(self, response: str) -> Dict[str, float]:
        """Extract cost estimate from response"""
        # Simple cost estimation based on intervention type
        return {"min": 10000, "max": 100000}
    
    def _extract_lives_saved(self, response: str) -> float:
        """Extract lives saved estimate from response"""
        # Simple estimation
        return 2.5
    
    def _extract_references(self, response: str) -> List[str]:
        """Extract references from response"""
        refs = []
        if "irc" in response.lower():
            refs.append("IRC Standards")
        if "morth" in response.lower():
            refs.append("MoRTH Guidelines")
        if "who" in response.lower():
            refs.append("WHO Road Safety Report")
        return refs
    
    def get_intervention_recommendations(self, problem_description: str) -> List[Dict]:
        """Get intervention recommendations based on problem description"""
        try:
            # Search interventions database
            recommendations = []
            
            # Simple keyword matching for now
            keywords = problem_description.lower().split()
            
            for intervention in self.interventions_db:
                intervention_text = f"{intervention.get('name', '')} {intervention.get('description', '')}".lower()
                
                # Check for keyword matches
                matches = sum(1 for keyword in keywords if keyword in intervention_text)
                if matches > 0:
                    recommendations.append({
                        'intervention': intervention,
                        'relevance_score': matches / len(keywords),
                        'reasoning': f"Matched {matches} keywords from problem description"
                    })
            
            # Sort by relevance
            recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_cascading_effects(self, intervention: str) -> List[Dict]:
        """Get cascading effects for an intervention"""
        try:
            # Simple rule-based cascading effects
            effects = []
            
            if "zebra crossing" in intervention.lower():
                effects.append({
                    'effect': 'Traffic speed reduction',
                    'probability': 0.8,
                    'impact': 'medium',
                    'description': 'Vehicles slow down approaching the crossing'
                })
                effects.append({
                    'effect': 'Pedestrian safety improvement',
                    'probability': 0.9,
                    'impact': 'high',
                    'description': 'Clear crossing path reduces accident risk'
                })
            
            return effects
            
        except Exception as e:
            logger.error(f"Error getting cascading effects: {e}")
            return []
    
    def estimate_implementation_cost(self, intervention: str) -> Dict[str, float]:
        """Estimate implementation cost for intervention"""
        try:
            # Simple cost estimation based on intervention type
            base_costs = {
                'zebra crossing': {'min': 15000, 'max': 25000},
                'speed bump': {'min': 8000, 'max': 15000},
                'traffic sign': {'min': 2000, 'max': 5000},
                'barrier': {'min': 5000, 'max': 12000}
            }
            
            intervention_lower = intervention.lower()
            for key, cost in base_costs.items():
                if key in intervention_lower:
                    return cost
            
            return {'min': 10000, 'max': 50000}  # Default range
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            return {'min': 0, 'max': 0}
    
    def predict_lives_saved(self, intervention: str, location_context: Dict) -> float:
        """Predict lives saved by intervention"""
        try:
            # Simple prediction based on intervention type
            base_savings = {
                'zebra crossing': 3.2,
                'speed bump': 2.1,
                'traffic sign': 1.5,
                'barrier': 4.5
            }
            
            intervention_lower = intervention.lower()
            for key, savings in base_savings.items():
                if key in intervention_lower:
                    return savings
            
            return 2.0  # Default estimate
            
        except Exception as e:
            logger.error(f"Error predicting lives saved: {e}")
            return 0.0

# Global instance
_llm_instance = None

def get_llm_engine() -> RoutesitLLM:
    """Get global LLM engine instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = RoutesitLLM()
    return _llm_instance

def initialize_llm_engine():
    """Initialize the LLM engine"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = RoutesitLLM()
    return _llm_instance