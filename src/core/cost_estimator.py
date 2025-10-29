import json
from typing import Dict, Any, List
from dataclasses import dataclass
import requests
from pathlib import Path

@dataclass
class CostEstimate:
    materials: float
    labor: float
    equipment: float
    overhead: float
    total: float
    currency: str = "INR"

class CPWDCostEstimator:
    """CPWD Schedule of Rates based cost estimation"""
    
    def __init__(self):
        self.sor_rates = self._load_sor_rates()
        self.location_multipliers = {
            'delhi': 1.0,
            'mumbai': 1.15,
            'bangalore': 1.08,
            'chennai': 1.05,
            'kolkata': 1.02,
            'hyderabad': 1.06,
            'pune': 1.10,
            'ahmedabad': 1.04,
            'default': 1.0
        }
    
    def _load_sor_rates(self) -> Dict[str, Any]:
        """Load CPWD SOR rates"""
        return {
            'road_signs': {
                'stop_sign': {'material': 2500, 'labor': 1500, 'equipment': 500},
                'speed_limit_sign': {'material': 2000, 'labor': 1200, 'equipment': 400},
                'warning_sign': {'material': 1800, 'labor': 1000, 'equipment': 300},
                'information_sign': {'material': 1500, 'labor': 800, 'equipment': 200}
            },
            'road_markings': {
                'zebra_crossing': {'material': 8000, 'labor': 4000, 'equipment': 2000},
                'lane_marking': {'material': 6000, 'labor': 3000, 'equipment': 1500},
                'arrow_marking': {'material': 2000, 'labor': 1000, 'equipment': 500},
                'stop_line': {'material': 3000, 'labor': 1500, 'equipment': 800}
            },
            'traffic_calming': {
                'speed_hump': {'material': 15000, 'labor': 8000, 'equipment': 3000},
                'rumble_strip': {'material': 8000, 'labor': 4000, 'equipment': 2000},
                'traffic_circle': {'material': 50000, 'labor': 25000, 'equipment': 10000}
            },
            'infrastructure': {
                'guard_rail': {'material': 20000, 'labor': 10000, 'equipment': 5000},
                'street_lighting': {'material': 30000, 'labor': 15000, 'equipment': 8000},
                'pedestrian_bridge': {'material': 400000, 'labor': 200000, 'equipment': 50000}
            }
        }
    
    def estimate_cost(self, intervention_type: str, intervention_name: str, 
                     location: str = 'default', quantity: int = 1) -> CostEstimate:
        """Estimate cost for intervention"""
        
        # Find matching SOR rate
        base_cost = self._find_sor_rate(intervention_type, intervention_name)
        
        if not base_cost:
            # Fallback estimation
            base_cost = self._estimate_fallback_cost(intervention_type, intervention_name)
        
        # Apply location multiplier
        location_mult = self.location_multipliers.get(location.lower(), 1.0)
        
        # Calculate costs
        materials = base_cost['material'] * quantity * location_mult
        labor = base_cost['labor'] * quantity * location_mult
        equipment = base_cost['equipment'] * quantity * location_mult
        overhead = (materials + labor + equipment) * 0.1  # 10% overhead
        
        total = materials + labor + equipment + overhead
        
        return CostEstimate(
            materials=materials,
            labor=labor,
            equipment=equipment,
            overhead=overhead,
            total=total
        )
    
    def _find_sor_rate(self, intervention_type: str, intervention_name: str) -> Dict[str, float]:
        """Find SOR rate for intervention"""
        
        type_rates = self.sor_rates.get(intervention_type.lower(), {})
        
        # Try exact match first
        if intervention_name.lower() in type_rates:
            return type_rates[intervention_name.lower()]
        
        # Try partial match
        for key, rate in type_rates.items():
            if key in intervention_name.lower() or intervention_name.lower() in key:
                return rate
        
        return None
    
    def _estimate_fallback_cost(self, intervention_type: str, intervention_name: str) -> Dict[str, float]:
        """Fallback cost estimation"""
        
        base_costs = {
            'road_sign': {'material': 2000, 'labor': 1200, 'equipment': 400},
            'road_marking': {'material': 5000, 'labor': 2500, 'equipment': 1000},
            'traffic_calming': {'material': 15000, 'labor': 8000, 'equipment': 3000},
            'infrastructure': {'material': 50000, 'labor': 25000, 'equipment': 10000}
        }
        
        return base_costs.get(intervention_type.lower(), {'material': 10000, 'labor': 5000, 'equipment': 2000})

def create_cost_estimation_system():
    """Create comprehensive cost estimation system"""
    estimator = CPWDCostEstimator()
    
    # Test the system
    test_cases = [
        ('road_sign', 'stop_sign', 'delhi', 1),
        ('road_marking', 'zebra_crossing', 'mumbai', 1),
        ('traffic_calming', 'speed_hump', 'bangalore', 2)
    ]
    
    results = []
    for case in test_cases:
        cost = estimator.estimate_cost(*case)
        results.append({
            'intervention': case[1],
            'location': case[2],
            'quantity': case[3],
            'cost_breakdown': {
                'materials': cost.materials,
                'labor': cost.labor,
                'equipment': cost.equipment,
                'overhead': cost.overhead,
                'total': cost.total
            }
        })
    
    return results

if __name__ == "__main__":
    results = create_cost_estimation_system()
    print(json.dumps(results, indent=2))
