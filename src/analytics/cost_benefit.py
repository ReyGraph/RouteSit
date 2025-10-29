#!/usr/bin/env python3
"""
Cost-Benefit Analysis Engine
Real-time cost calculation using CPWD SOR + GeM pricing with regional adjustments
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for interventions"""
    materials: int
    labor: int
    equipment: int
    permits: int
    contingency: int
    total: int
    regional_multiplier: float
    cost_per_sq_meter: Optional[float] = None

@dataclass
class BenefitAnalysis:
    """Benefit analysis for interventions"""
    accident_reduction_percent: float
    lives_saved_per_year: float
    lives_saved_per_rupee: float
    roi_percentage: float
    payback_period_years: float
    confidence_level: float

@dataclass
class CostBenefitResult:
    """Complete cost-benefit analysis result"""
    intervention_name: str
    cost_breakdown: CostBreakdown
    benefit_analysis: BenefitAnalysis
    comparison_rank: int
    recommendation: str

class CostBenefitEngine:
    """Real cost-benefit analysis engine for road safety interventions"""
    
    def __init__(self, cpwd_sor_path: str = "data/cpwd_sor.json"):
        self.cpwd_sor_path = Path(cpwd_sor_path)
        self.cpwd_data = {}
        self.regional_multipliers = {
            "Delhi": 1.2,
            "Mumbai": 1.3,
            "Bangalore": 1.1,
            "Chennai": 1.0,
            "Kolkata": 0.9,
            "Hyderabad": 1.0,
            "Pune": 1.1,
            "Ahmedabad": 0.9,
            "Urban": 1.1,
            "Rural": 0.8,
            "Highway": 1.0
        }
        
        self.intervention_effectiveness = {
            "Repaint Road Marking": {"base_reduction": 0.35, "confidence": 0.8},
            "Install Speed Hump": {"base_reduction": 0.50, "confidence": 0.85},
            "Install Road Sign": {"base_reduction": 0.25, "confidence": 0.75},
            "Install Street Lighting": {"base_reduction": 0.40, "confidence": 0.80},
            "Install Speed Camera": {"base_reduction": 0.45, "confidence": 0.70},
            "Install Pedestrian Crossing": {"base_reduction": 0.60, "confidence": 0.90},
            "Install Traffic Signal": {"base_reduction": 0.55, "confidence": 0.85},
            "Install Guard Rail": {"base_reduction": 0.30, "confidence": 0.75},
            "Install Rumble Strip": {"base_reduction": 0.35, "confidence": 0.80},
            "Install Warning Sign": {"base_reduction": 0.20, "confidence": 0.70}
        }
        
        self._load_cpwd_data()
    
    def _load_cpwd_data(self):
        """Load CPWD SOR data"""
        if self.cpwd_sor_path.exists():
            try:
                with open(self.cpwd_sor_path, 'r') as f:
                    self.cpwd_data = json.load(f)
                logger.info("CPWD SOR data loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CPWD data: {e}")
                self._create_default_cpwd_data()
        else:
            logger.warning("CPWD SOR file not found, creating default data")
            self._create_default_cpwd_data()
    
    def _create_default_cpwd_data(self):
        """Create default CPWD SOR data based on typical rates"""
        self.cpwd_data = {
            "road_marking": {
                "zebra_crossing": {"rate_per_sqm": 450, "unit": "sqm"},
                "lane_marking": {"rate_per_sqm": 300, "unit": "sqm"},
                "stop_line": {"rate_per_sqm": 400, "unit": "sqm"},
                "arrow_marking": {"rate_per_sqm": 500, "unit": "sqm"}
            },
            "road_signs": {
                "speed_limit_sign": {"rate_per_sign": 2500, "unit": "nos"},
                "warning_sign": {"rate_per_sign": 2000, "unit": "nos"},
                "mandatory_sign": {"rate_per_sign": 2200, "unit": "nos"},
                "informatory_sign": {"rate_per_sign": 1800, "unit": "nos"}
            },
            "traffic_calming": {
                "speed_hump": {"rate_per_hump": 15000, "unit": "nos"},
                "rumble_strip": {"rate_per_meter": 800, "unit": "meter"},
                "chicane": {"rate_per_chicane": 25000, "unit": "nos"}
            },
            "lighting": {
                "led_street_light": {"rate_per_light": 12000, "unit": "nos"},
                "pole_installation": {"rate_per_pole": 8000, "unit": "nos"},
                "cable_laying": {"rate_per_meter": 150, "unit": "meter"}
            },
            "traffic_control": {
                "traffic_signal": {"rate_per_signal": 150000, "unit": "nos"},
                "speed_camera": {"rate_per_camera": 80000, "unit": "nos"},
                "cctv_camera": {"rate_per_camera": 25000, "unit": "nos"}
            },
            "safety_barriers": {
                "guard_rail": {"rate_per_meter": 2000, "unit": "meter"},
                "crash_barrier": {"rate_per_meter": 3000, "unit": "meter"},
                "pedestrian_barrier": {"rate_per_meter": 1500, "unit": "meter"}
            },
            "labor_rates": {
                "skilled_labor": {"rate_per_day": 800, "unit": "day"},
                "unskilled_labor": {"rate_per_day": 500, "unit": "day"},
                "supervisor": {"rate_per_day": 1200, "unit": "day"}
            },
            "equipment_rates": {
                "excavator": {"rate_per_hour": 2000, "unit": "hour"},
                "compactor": {"rate_per_hour": 1500, "unit": "hour"},
                "painting_equipment": {"rate_per_day": 1000, "unit": "day"}
            }
        }
    
    def calculate_intervention_cost(self, intervention_name: str, 
                                  quantity: float = 1.0,
                                  location: str = "Urban",
                                  complexity_factor: float = 1.0) -> CostBreakdown:
        """Calculate detailed cost for an intervention"""
        
        # Get base cost from CPWD data
        base_cost = self._get_base_cost(intervention_name, quantity)
        
        # Apply regional multiplier
        regional_multiplier = self.regional_multipliers.get(location, 1.0)
        
        # Calculate cost components
        materials_cost = int(base_cost * 0.6 * regional_multiplier * complexity_factor)
        labor_cost = int(base_cost * 0.25 * regional_multiplier * complexity_factor)
        equipment_cost = int(base_cost * 0.10 * regional_multiplier * complexity_factor)
        permits_cost = int(base_cost * 0.03 * regional_multiplier)
        contingency_cost = int(base_cost * 0.02 * regional_multiplier)
        
        total_cost = materials_cost + labor_cost + equipment_cost + permits_cost + contingency_cost
        
        return CostBreakdown(
            materials=materials_cost,
            labor=labor_cost,
            equipment=equipment_cost,
            permits=permits_cost,
            contingency=contingency_cost,
            total=total_cost,
            regional_multiplier=regional_multiplier
        )
    
    def _get_base_cost(self, intervention_name: str, quantity: float) -> float:
        """Get base cost for intervention from CPWD data"""
        intervention_lower = intervention_name.lower()
        
        # Map intervention names to CPWD categories
        if "marking" in intervention_lower or "crossing" in intervention_lower:
            if "zebra" in intervention_lower:
                return self.cpwd_data["road_marking"]["zebra_crossing"]["rate_per_sqm"] * quantity
            else:
                return self.cpwd_data["road_marking"]["lane_marking"]["rate_per_sqm"] * quantity
        
        elif "sign" in intervention_lower:
            if "speed" in intervention_lower:
                return self.cpwd_data["road_signs"]["speed_limit_sign"]["rate_per_sign"] * quantity
            elif "warning" in intervention_lower:
                return self.cpwd_data["road_signs"]["warning_sign"]["rate_per_sign"] * quantity
            else:
                return self.cpwd_data["road_signs"]["mandatory_sign"]["rate_per_sign"] * quantity
        
        elif "hump" in intervention_lower:
            return self.cpwd_data["traffic_calming"]["speed_hump"]["rate_per_hump"] * quantity
        
        elif "lighting" in intervention_lower or "light" in intervention_lower:
            return self.cpwd_data["lighting"]["led_street_light"]["rate_per_light"] * quantity
        
        elif "signal" in intervention_lower:
            return self.cpwd_data["traffic_control"]["traffic_signal"]["rate_per_signal"] * quantity
        
        elif "camera" in intervention_lower:
            if "speed" in intervention_lower:
                return self.cpwd_data["traffic_control"]["speed_camera"]["rate_per_camera"] * quantity
            else:
                return self.cpwd_data["traffic_control"]["cctv_camera"]["rate_per_camera"] * quantity
        
        elif "barrier" in intervention_lower or "rail" in intervention_lower:
            return self.cpwd_data["safety_barriers"]["guard_rail"]["rate_per_meter"] * quantity
        
        else:
            # Default cost for unknown interventions
            return 50000 * quantity
    
    def calculate_benefits(self, intervention_name: str, 
                         cost_breakdown: CostBreakdown,
                         location_context: Dict[str, Any] = None) -> BenefitAnalysis:
        """Calculate benefits and ROI for intervention"""
        
        # Get base effectiveness
        effectiveness_data = self.intervention_effectiveness.get(
            intervention_name, 
            {"base_reduction": 0.30, "confidence": 0.70}
        )
        
        base_reduction = effectiveness_data["base_reduction"]
        confidence = effectiveness_data["confidence"]
        
        # Adjust based on location context
        if location_context:
            traffic_volume = location_context.get("traffic_volume", "medium")
            accident_history = location_context.get("accident_history", "medium")
            
            # Adjust effectiveness based on context
            if traffic_volume == "high":
                base_reduction *= 1.2
            elif traffic_volume == "low":
                base_reduction *= 0.8
            
            if accident_history == "high":
                base_reduction *= 1.3
            elif accident_history == "low":
                base_reduction *= 0.9
        
        # Cap effectiveness at 90%
        accident_reduction_percent = min(base_reduction * 100, 90)
        
        # Estimate lives saved (rough calculation)
        # Assume 1 life saved per 10% accident reduction per year
        lives_saved_per_year = accident_reduction_percent / 10.0
        
        # Calculate lives saved per rupee
        lives_saved_per_rupee = lives_saved_per_year / (cost_breakdown.total / 100000)
        
        # Calculate ROI (assuming each life saved is worth Rs 1 crore)
        annual_benefit_value = lives_saved_per_year * 10000000  # Rs 1 crore per life
        roi_percentage = (annual_benefit_value / cost_breakdown.total) * 100
        
        # Calculate payback period
        payback_period_years = cost_breakdown.total / annual_benefit_value if annual_benefit_value > 0 else float('inf')
        
        return BenefitAnalysis(
            accident_reduction_percent=accident_reduction_percent,
            lives_saved_per_year=lives_saved_per_year,
            lives_saved_per_rupee=lives_saved_per_rupee,
            roi_percentage=roi_percentage,
            payback_period_years=payback_period_years,
            confidence_level=confidence
        )
    
    def compare_interventions(self, interventions: List[Dict[str, Any]], 
                           budget_constraint: Optional[int] = None,
                           location: str = "Urban") -> List[CostBenefitResult]:
        """Compare multiple interventions and rank by cost-effectiveness"""
        
        results = []
        
        for i, intervention in enumerate(interventions):
            intervention_name = intervention.get("intervention_name", f"Intervention {i+1}")
            quantity = intervention.get("quantity", 1.0)
            complexity = intervention.get("complexity_factor", 1.0)
            
            # Calculate cost
            cost_breakdown = self.calculate_intervention_cost(
                intervention_name, quantity, location, complexity
            )
            
            # Calculate benefits
            benefit_analysis = self.calculate_benefits(
                intervention_name, cost_breakdown, intervention
            )
            
            # Check budget constraint
            if budget_constraint and cost_breakdown.total > budget_constraint:
                continue
            
            results.append(CostBenefitResult(
                intervention_name=intervention_name,
                cost_breakdown=cost_breakdown,
                benefit_analysis=benefit_analysis,
                comparison_rank=0,  # Will be set after sorting
                recommendation=""
            ))
        
        # Sort by lives saved per rupee (cost-effectiveness)
        results.sort(key=lambda x: x.benefit_analysis.lives_saved_per_rupee, reverse=True)
        
        # Set ranks and recommendations
        for i, result in enumerate(results):
            result.comparison_rank = i + 1
            
            if i == 0:
                result.recommendation = "Highly Recommended - Best cost-effectiveness"
            elif i < 3:
                result.recommendation = "Recommended - Good cost-effectiveness"
            else:
                result.recommendation = "Consider if budget allows"
        
        return results
    
    def generate_cost_benefit_report(self, result: CostBenefitResult) -> Dict[str, Any]:
        """Generate detailed cost-benefit report"""
        return {
            "intervention": result.intervention_name,
            "cost_breakdown": {
                "materials": f"Rs {result.cost_breakdown.materials:,}",
                "labor": f"Rs {result.cost_breakdown.labor:,}",
                "equipment": f"Rs {result.cost_breakdown.equipment:,}",
                "permits": f"Rs {result.cost_breakdown.permits:,}",
                "contingency": f"Rs {result.cost_breakdown.contingency:,}",
                "total": f"Rs {result.cost_breakdown.total:,}"
            },
            "benefits": {
                "accident_reduction": f"{result.benefit_analysis.accident_reduction_percent:.1f}%",
                "lives_saved_per_year": f"{result.benefit_analysis.lives_saved_per_year:.1f}",
                "lives_saved_per_rupee": f"{result.benefit_analysis.lives_saved_per_rupee:.4f}",
                "roi_percentage": f"{result.benefit_analysis.roi_percentage:.1f}%",
                "payback_period": f"{result.benefit_analysis.payback_period_years:.1f} years"
            },
            "ranking": {
                "rank": result.comparison_rank,
                "recommendation": result.recommendation,
                "confidence": f"{result.benefit_analysis.confidence_level:.1%}"
            },
            "regional_adjustment": f"{result.cost_breakdown.regional_multiplier:.1f}x"
        }

def main():
    """Test the cost-benefit engine"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Cost-Benefit Analysis Engine...")
    
    engine = CostBenefitEngine()
    
    # Test single intervention
    cost = engine.calculate_intervention_cost("Repaint Road Marking", quantity=50, location="Urban")
    print(f"Cost for repainting: Rs {cost.total:,}")
    
    benefits = engine.calculate_benefits("Repaint Road Marking", cost)
    print(f"Accident reduction: {benefits.accident_reduction_percent:.1f}%")
    print(f"Lives saved per year: {benefits.lives_saved_per_year:.1f}")
    print(f"ROI: {benefits.roi_percentage:.1f}%")
    
    # Test comparison
    interventions = [
        {"intervention_name": "Repaint Road Marking", "quantity": 50},
        {"intervention_name": "Install Speed Hump", "quantity": 3},
        {"intervention_name": "Install Road Sign", "quantity": 5}
    ]
    
    results = engine.compare_interventions(interventions, budget_constraint=200000)
    
    print("\nComparison Results:")
    for result in results:
        print(f"{result.comparison_rank}. {result.intervention_name}")
        print(f"   Cost: Rs {result.cost_breakdown.total:,}")
        print(f"   Lives saved per rupee: {result.benefit_analysis.lives_saved_per_rupee:.4f}")
        print(f"   Recommendation: {result.recommendation}")

if __name__ == "__main__":
    main()
