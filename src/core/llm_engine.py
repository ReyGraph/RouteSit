import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Any, Optional
import logging
import warnings

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class LocalLLMEngine:
    """Local LLM engine for reasoning and explanation generation"""
    
    def __init__(self):
        self.model_name = config.get('model.llm.name', 'meta-llama/Llama-2-7b-chat-hf')
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the local LLM model"""
        try:
            logger.info(f"Initializing local LLM: {self.model_name}")
            
            # Suppress warnings
            warnings.filterwarnings("ignore")
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if config.get('model.llm.quantized', True) and device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=config.get('model.llm.quantization_config.load_in_4bit', True),
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type=config.get('model.llm.quantization_config.bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_use_double_quant=config.get('model.llm.quantization_config.bnb_4bit_use_double_quant', True)
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if device == "cuda" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if device == "cuda" else None,
                max_length=config.get('model.llm.max_length', 2048),
                temperature=config.get('model.llm.temperature', 0.7),
                top_p=config.get('model.llm.top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Local LLM initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            # Fallback to a lighter model or mock implementation
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback implementation for when LLM fails to load"""
        logger.warning("Using fallback LLM implementation")
        self.model_name = "fallback"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
    
    def generate_reasoning(self, query: str, interventions: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """
        Generate reasoning for intervention recommendations
        
        Args:
            query: User's problem description
            interventions: List of relevant interventions
            context: Additional context (road type, traffic, etc.)
            
        Returns:
            Generated reasoning text
        """
        try:
            if self.pipeline is None:
                return self._generate_fallback_reasoning(query, interventions, context)
            
            # Prepare prompt
            prompt = self._create_reasoning_prompt(query, interventions, context)
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            reasoning = generated_text[len(prompt):].strip()
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return self._generate_fallback_reasoning(query, interventions, context)
    
    def _create_reasoning_prompt(self, query: str, interventions: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Create prompt for reasoning generation"""
        
        # Format interventions
        interventions_text = ""
        for i, intervention in enumerate(interventions[:5], 1):  # Limit to top 5
            interventions_text += f"{i}. {intervention['intervention_name']}\n"
            interventions_text += f"   Cost: ₹{intervention['cost_estimate']['total']:,}\n"
            interventions_text += f"   Impact: {intervention['predicted_impact']['accident_reduction_percent']}% accident reduction\n"
            interventions_text += f"   Description: {intervention['description']}\n\n"
        
        # Format context
        context_text = ""
        if context:
            context_text = f"Additional Context:\n"
            for key, value in context.items():
                context_text += f"- {key}: {value}\n"
            context_text += "\n"
        
        prompt = f"""You are a road safety expert AI assistant. Analyze the following road safety problem and provide reasoning for the recommended interventions.

Problem Description: {query}

{context_text}Recommended Interventions:
{interventions_text}

Please provide detailed reasoning for why these interventions are recommended, considering:
1. How each intervention addresses the specific problem
2. Cost-effectiveness analysis
3. Implementation feasibility
4. Expected safety impact
5. Any dependencies or conflicts between interventions

Reasoning:"""
        
        return prompt
    
    def _generate_fallback_reasoning(self, query: str, interventions: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Generate fallback reasoning when LLM is not available"""
        
        reasoning_parts = []
        
        # Problem analysis
        reasoning_parts.append(f"Problem Analysis: The issue described as '{query}' requires immediate attention to improve road safety.")
        
        # Intervention analysis
        if interventions:
            reasoning_parts.append(f"\nRecommended Interventions Analysis:")
            
            for i, intervention in enumerate(interventions[:3], 1):
                name = intervention['intervention_name']
                cost = intervention['cost_estimate']['total']
                impact = intervention['predicted_impact']['accident_reduction_percent']
                category = intervention['category']
                
                reasoning_parts.append(f"\n{i}. {name}")
                reasoning_parts.append(f"   - Category: {category.replace('_', ' ').title()}")
                reasoning_parts.append(f"   - Cost: ₹{cost:,}")
                reasoning_parts.append(f"   - Expected Impact: {impact}% reduction in accidents")
                reasoning_parts.append(f"   - Cost-Effectiveness: ₹{cost//impact:,} per percentage point of improvement")
        
        # Cost-benefit analysis
        if interventions:
            total_cost = sum(intv['cost_estimate']['total'] for intv in interventions[:3])
            avg_impact = sum(intv['predicted_impact']['accident_reduction_percent'] for intv in interventions[:3]) / len(interventions[:3])
            
            reasoning_parts.append(f"\nCost-Benefit Analysis:")
            reasoning_parts.append(f"- Total Implementation Cost: ₹{total_cost:,}")
            reasoning_parts.append(f"- Average Safety Improvement: {avg_impact:.1f}%")
            reasoning_parts.append(f"- Cost per Life Saved: ₹{total_cost//max(1, sum(intv['predicted_impact']['lives_saved_per_year'] for intv in interventions[:3])):,.0f}")
        
        # Implementation considerations
        reasoning_parts.append(f"\nImplementation Considerations:")
        reasoning_parts.append("- All interventions comply with IRC and MoRTH standards")
        reasoning_parts.append("- Site survey and traffic analysis required before implementation")
        reasoning_parts.append("- Local authority approval needed")
        reasoning_parts.append("- Regular maintenance schedule should be established")
        
        return "\n".join(reasoning_parts)
    
    def generate_explanation(self, intervention: Dict[str, Any], problem_context: str) -> str:
        """Generate detailed explanation for a specific intervention"""
        
        try:
            if self.pipeline is None:
                return self._generate_fallback_explanation(intervention, problem_context)
            
            prompt = f"""Explain why this road safety intervention is recommended for the given problem:

Problem: {problem_context}

Intervention: {intervention['intervention_name']}
Description: {intervention['description']}
Cost: ₹{intervention['cost_estimate']['total']:,}
Expected Impact: {intervention['predicted_impact']['accident_reduction_percent']}% accident reduction

Provide a clear explanation covering:
1. How this intervention solves the problem
2. Why it's cost-effective
3. Implementation requirements
4. Expected outcomes

Explanation:"""
            
            response = self.pipeline(
                prompt,
                max_new_tokens=300,
                temperature=0.6,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            explanation = generated_text[len(prompt):].strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return self._generate_fallback_explanation(intervention, problem_context)
    
    def _generate_fallback_explanation(self, intervention: Dict[str, Any], problem_context: str) -> str:
        """Generate fallback explanation"""
        
        explanation_parts = []
        
        explanation_parts.append(f"Intervention: {intervention['intervention_name']}")
        explanation_parts.append(f"Problem Context: {problem_context}")
        explanation_parts.append(f"\nWhy This Intervention:")
        
        # Problem-solving rationale
        problem_type = intervention.get('problem_type', '')
        category = intervention.get('category', '')
        
        if 'faded' in problem_type:
            explanation_parts.append("- Addresses visibility issues caused by faded markings/signs")
        elif 'missing' in problem_type:
            explanation_parts.append("- Fills critical safety gap where no intervention exists")
        elif 'damaged' in problem_type:
            explanation_parts.append("- Restores safety functionality of damaged infrastructure")
        
        # Category-specific benefits
        if category == 'road_sign':
            explanation_parts.append("- Improves driver awareness and compliance")
            explanation_parts.append("- Provides clear guidance for traffic flow")
        elif category == 'road_marking':
            explanation_parts.append("- Enhances road visibility and lane discipline")
            explanation_parts.append("- Guides pedestrian and vehicle movement")
        elif category == 'traffic_calming':
            explanation_parts.append("- Reduces vehicle speeds in critical areas")
            explanation_parts.append("- Improves pedestrian safety")
        elif category == 'infrastructure':
            explanation_parts.append("- Provides physical safety barriers")
            explanation_parts.append("- Improves overall road infrastructure")
        
        # Cost-effectiveness
        cost = intervention['cost_estimate']['total']
        impact = intervention['predicted_impact']['accident_reduction_percent']
        explanation_parts.append(f"\nCost-Effectiveness:")
        explanation_parts.append(f"- Implementation Cost: ₹{cost:,}")
        explanation_parts.append(f"- Expected Safety Improvement: {impact}%")
        explanation_parts.append(f"- Cost per Percentage Point: ₹{cost//max(1, impact):,}")
        
        # Implementation details
        timeline = intervention.get('implementation_timeline', 1)
        explanation_parts.append(f"\nImplementation:")
        explanation_parts.append(f"- Timeline: {timeline} day{'s' if timeline > 1 else ''}")
        explanation_parts.append("- Complies with IRC and MoRTH standards")
        explanation_parts.append("- Requires local authority approval")
        
        return "\n".join(explanation_parts)
    
    def generate_summary(self, query: str, interventions: List[Dict[str, Any]], recommendations: Dict[str, Any]) -> str:
        """Generate executive summary of recommendations"""
        
        try:
            if self.pipeline is None:
                return self._generate_fallback_summary(query, interventions, recommendations)
            
            prompt = f"""Generate an executive summary for road safety intervention recommendations:

Problem: {query}

Number of Recommended Interventions: {len(interventions)}
Total Estimated Cost: ₹{sum(intv['cost_estimate']['total'] for intv in interventions):,}
Average Safety Improvement: {sum(intv['predicted_impact']['accident_reduction_percent'] for intv in interventions) / len(interventions):.1f}%

Provide a concise executive summary covering:
1. Problem assessment
2. Recommended solution approach
3. Expected outcomes
4. Implementation timeline
5. Cost-benefit summary

Executive Summary:"""
            
            response = self.pipeline(
                prompt,
                max_new_tokens=400,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            summary = generated_text[len(prompt):].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return self._generate_fallback_summary(query, interventions, recommendations)
    
    def _generate_fallback_summary(self, query: str, interventions: List[Dict[str, Any]], recommendations: Dict[str, Any]) -> str:
        """Generate fallback summary"""
        
        summary_parts = []
        
        summary_parts.append("EXECUTIVE SUMMARY")
        summary_parts.append("=" * 50)
        
        summary_parts.append(f"\nProblem: {query}")
        
        if interventions:
            total_cost = sum(intv['cost_estimate']['total'] for intv in interventions)
            avg_impact = sum(intv['predicted_impact']['accident_reduction_percent'] for intv in interventions) / len(interventions)
            total_lives_saved = sum(intv['predicted_impact']['lives_saved_per_year'] for intv in interventions)
            
            summary_parts.append(f"\nRecommended Solution:")
            summary_parts.append(f"- {len(interventions)} interventions identified")
            summary_parts.append(f"- Total implementation cost: ₹{total_cost:,}")
            summary_parts.append(f"- Average safety improvement: {avg_impact:.1f}%")
            summary_parts.append(f"- Estimated lives saved per year: {total_lives_saved:.1f}")
            
            summary_parts.append(f"\nExpected Outcomes:")
            summary_parts.append(f"- Significant reduction in road accidents")
            summary_parts.append(f"- Improved traffic flow and safety")
            summary_parts.append(f"- Enhanced compliance with traffic regulations")
            summary_parts.append(f"- Better pedestrian and cyclist safety")
            
            summary_parts.append(f"\nImplementation Timeline:")
            max_timeline = max(intv.get('implementation_timeline', 1) for intv in interventions)
            summary_parts.append(f"- Estimated completion: {max_timeline} days")
            summary_parts.append(f"- Phased implementation recommended")
            
            summary_parts.append(f"\nCost-Benefit Analysis:")
            summary_parts.append(f"- Cost per life saved: ₹{total_cost//max(1, total_lives_saved):,.0f}")
            summary_parts.append(f"- Return on investment: High safety impact")
            summary_parts.append(f"- Long-term maintenance costs: Minimal")
        
        return "\n".join(summary_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'is_quantized': config.get('model.llm.quantized', True),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': config.get('model.llm.max_length', 2048),
            'temperature': config.get('model.llm.temperature', 0.7)
        }
