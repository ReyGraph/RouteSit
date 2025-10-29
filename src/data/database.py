import json
import random
from typing import List, Dict, Any
from pathlib import Path

class InterventionDatabase:
    """Comprehensive intervention database for road safety"""
    
    def __init__(self, data_dir: str = "data/interventions"):
        self.data_dir = Path(data_dir)
        self.interventions = []
        self._load_interventions()
    
    def _load_interventions(self):
        """Load interventions from JSON files"""
        intervention_file = self.data_dir / "interventions.json"
        if intervention_file.exists():
            with open(intervention_file, 'r', encoding='utf-8') as f:
                self.interventions = json.load(f)
        else:
            self.interventions = []
    
    def save_interventions(self):
        """Save interventions to JSON file"""
        intervention_file = self.data_dir / "interventions.json"
        with open(intervention_file, 'w', encoding='utf-8') as f:
            json.dump(self.interventions, f, indent=2, ensure_ascii=False)
    
    def generate_comprehensive_database(self):
        """Generate comprehensive intervention database with 300+ entries"""
        
        # Road Signs Category (100+ interventions)
        road_signs = self._generate_road_sign_interventions()
        
        # Road Markings Category (80+ interventions)
        road_markings = self._generate_road_marking_interventions()
        
        # Traffic Calming Category (60+ interventions)
        traffic_calming = self._generate_traffic_calming_interventions()
        
        # Infrastructure Category (60+ interventions)
        infrastructure = self._generate_infrastructure_interventions()
        
        # Combine all interventions
        self.interventions = road_signs + road_markings + traffic_calming + infrastructure
        
        # Shuffle to randomize order
        random.shuffle(self.interventions)
        
        # Assign sequential IDs
        for i, intervention in enumerate(self.interventions):
            intervention['intervention_id'] = f"int_{i+1:03d}"
        
        return len(self.interventions)
    
    def _generate_road_sign_interventions(self) -> List[Dict[str, Any]]:
        """Generate road sign interventions"""
        interventions = []
        
        # Regulatory Signs
        regulatory_signs = [
            ("STOP Sign", "stop", 4000, 45, "Replace damaged STOP sign with IRC-compliant installation"),
            ("YIELD Sign", "yield", 3500, 40, "Install YIELD sign for right-of-way control"),
            ("Speed Limit 30", "speed_limit", 2800, 35, "Install 30 kmph speed limit sign"),
            ("Speed Limit 50", "speed_limit", 2800, 30, "Install 50 kmph speed limit sign"),
            ("No Entry Sign", "no_entry", 3200, 50, "Install NO ENTRY sign for one-way control"),
            ("No Parking Sign", "no_parking", 2500, 25, "Install NO PARKING sign"),
            ("No Overtaking Sign", "no_overtaking", 3000, 30, "Install NO OVERTAKING sign"),
            ("No Horn Sign", "no_horn", 2000, 20, "Install NO HORN sign near hospitals/schools"),
            ("No U-Turn Sign", "no_uturn", 2800, 25, "Install NO U-TURN sign"),
            ("Compulsory Turn Left", "turn_left", 3000, 30, "Install compulsory left turn sign"),
            ("Compulsory Turn Right", "turn_right", 3000, 30, "Install compulsory right turn sign"),
            ("One Way Sign", "one_way", 3500, 40, "Install ONE WAY sign"),
            ("Load Limit Sign", "load_limit", 4000, 35, "Install load limit restriction sign"),
            ("Height Limit Sign", "height_limit", 3500, 30, "Install height restriction sign"),
            ("Width Limit Sign", "width_limit", 3500, 30, "Install width restriction sign")
        ]
        
        for sign_name, problem_type, cost, impact, description in regulatory_signs:
            interventions.extend(self._create_sign_variations(sign_name, problem_type, cost, impact, description))
        
        # Warning Signs
        warning_signs = [
            ("Sharp Curve Ahead", "curve_warning", 3500, 40, "Install sharp curve warning sign"),
            ("Steep Descent", "descent_warning", 3000, 35, "Install steep descent warning sign"),
            ("School Zone Ahead", "school_warning", 4000, 50, "Install school zone warning sign"),
            ("Hospital Zone", "hospital_warning", 4000, 45, "Install hospital zone warning sign"),
            ("Railway Crossing", "railway_warning", 5000, 60, "Install railway crossing warning sign"),
            ("Animal Crossing", "animal_warning", 3000, 30, "Install animal crossing warning sign"),
            ("Falling Rocks", "rocks_warning", 3500, 35, "Install falling rocks warning sign"),
            ("Slippery Road", "slippery_warning", 3000, 30, "Install slippery road warning sign"),
            ("Narrow Bridge", "bridge_warning", 4000, 40, "Install narrow bridge warning sign"),
            ("Road Work Ahead", "roadwork_warning", 3000, 25, "Install road work warning sign"),
            ("Pedestrian Crossing", "pedestrian_warning", 3500, 45, "Install pedestrian crossing warning sign"),
            ("Children Playing", "children_warning", 3000, 40, "Install children playing warning sign"),
            ("Cycle Crossing", "cycle_warning", 3000, 35, "Install cycle crossing warning sign"),
            ("T-Junction Ahead", "junction_warning", 3500, 40, "Install T-junction warning sign"),
            ("Y-Junction Ahead", "junction_warning", 3500, 40, "Install Y-junction warning sign")
        ]
        
        for sign_name, problem_type, cost, impact, description in warning_signs:
            interventions.extend(self._create_sign_variations(sign_name, problem_type, cost, impact, description))
        
        # Information Signs
        info_signs = [
            ("Direction to City", "direction", 3000, 20, "Install direction sign to nearest city"),
            ("Distance Marker", "distance", 2000, 15, "Install distance marker sign"),
            ("Rest Area", "rest_area", 4000, 25, "Install rest area information sign"),
            ("Fuel Station", "fuel_station", 3000, 20, "Install fuel station direction sign"),
            ("Hospital Direction", "hospital_direction", 3500, 30, "Install hospital direction sign"),
            ("Police Station", "police_direction", 3000, 25, "Install police station direction sign"),
            ("Tourist Information", "tourist_info", 4000, 20, "Install tourist information sign"),
            ("Emergency Contact", "emergency_contact", 3000, 25, "Install emergency contact information sign")
        ]
        
        for sign_name, problem_type, cost, impact, description in info_signs:
            interventions.extend(self._create_sign_variations(sign_name, problem_type, cost, impact, description))
        
        return interventions
    
    def _create_sign_variations(self, sign_name: str, problem_type: str, base_cost: int, base_impact: int, description: str) -> List[Dict[str, Any]]:
        """Create variations of a sign intervention for different problem types"""
        variations = []
        
        problem_types = ["damaged", "faded", "missing", "height_issue", "placement_error"]
        
        for prob_type in problem_types:
            cost_multiplier = {
                "damaged": 1.0,
                "faded": 0.8,
                "missing": 1.2,
                "height_issue": 1.5,
                "placement_error": 1.3
            }
            
            impact_multiplier = {
                "damaged": 1.0,
                "faded": 0.9,
                "missing": 1.1,
                "height_issue": 0.8,
                "placement_error": 0.7
            }
            
            intervention = {
                "problem_type": prob_type,
                "category": "road_sign",
                "intervention_name": f"{sign_name} - {prob_type.replace('_', ' ').title()}",
                "description": f"{description} addressing {prob_type.replace('_', ' ')} issue",
                "cost_estimate": {
                    "materials": int(base_cost * cost_multiplier[prob_type] * 0.6),
                    "labor": int(base_cost * cost_multiplier[prob_type] * 0.4),
                    "total": int(base_cost * cost_multiplier[prob_type]),
                    "currency": "INR"
                },
                "predicted_impact": {
                    "accident_reduction_percent": int(base_impact * impact_multiplier[prob_type]),
                    "confidence_level": "high" if prob_type in ["damaged", "missing"] else "medium",
                    "lives_saved_per_year": round(base_impact * impact_multiplier[prob_type] / 20, 1),
                    "injury_prevention_per_year": round(base_impact * impact_multiplier[prob_type] / 5, 1)
                },
                "implementation_timeline": 1 if prob_type == "faded" else 2,
                "references": [
                    {
                        "standard": "IRC67-2022",
                        "clause": "14.4",
                        "page": 156,
                        "description": f"{sign_name} specifications and placement requirements"
                    }
                ],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "prerequisites": {
                    "site_survey": True,
                    "traffic_count": prob_type == "missing",
                    "visibility_assessment": True
                },
                "compliance_requirements": [
                    "IRC 67-2022 compliance",
                    "MoRTH approval",
                    "Local authority permission"
                ],
                "maintenance_requirements": {
                    "inspection_frequency": "monthly",
                    "replacement_cycle": "5_years",
                    "cleaning_frequency": "weekly"
                }
            }
            
            variations.append(intervention)
        
        return variations
    
    def _generate_road_marking_interventions(self) -> List[Dict[str, Any]]:
        """Generate road marking interventions"""
        interventions = []
        
        marking_types = [
            ("Zebra Crossing", "zebra_crossing", 15000, 50, "Paint zebra crossing with thermoplastic material"),
            ("Stop Line", "stop_line", 8000, 40, "Paint stop line at intersection"),
            ("Give Way Line", "give_way_line", 6000, 35, "Paint give way line"),
            ("Center Line", "center_line", 12000, 30, "Paint center line separation"),
            ("Edge Line", "edge_line", 10000, 25, "Paint edge line marking"),
            ("Lane Marking", "lane_marking", 15000, 35, "Paint lane separation markings"),
            ("Arrow Marking", "arrow_marking", 5000, 30, "Paint directional arrow markings"),
            ("Speed Hump Marking", "speed_hump_marking", 3000, 25, "Paint speed hump warning markings"),
            ("Parking Bay Marking", "parking_marking", 8000, 20, "Paint parking bay markings"),
            ("Bus Stop Marking", "bus_stop_marking", 10000, 30, "Paint bus stop area markings"),
            ("School Zone Marking", "school_zone_marking", 12000, 45, "Paint school zone markings"),
            ("Hospital Zone Marking", "hospital_zone_marking", 12000, 40, "Paint hospital zone markings"),
            ("Cycle Lane Marking", "cycle_lane_marking", 18000, 40, "Paint dedicated cycle lane markings"),
            ("Pedestrian Walkway", "pedestrian_walkway", 20000, 45, "Paint pedestrian walkway markings"),
            ("No Parking Zone", "no_parking_zone", 8000, 25, "Paint no parking zone markings")
        ]
        
        for marking_name, problem_type, base_cost, base_impact, description in marking_types:
            interventions.extend(self._create_marking_variations(marking_name, problem_type, base_cost, base_impact, description))
        
        return interventions
    
    def _create_marking_variations(self, marking_name: str, problem_type: str, base_cost: int, base_impact: int, description: str) -> List[Dict[str, Any]]:
        """Create variations of marking interventions"""
        variations = []
        
        problem_types = ["faded", "missing", "damaged", "non_compliant", "placement_error"]
        
        for prob_type in problem_types:
            cost_multiplier = {
                "faded": 0.7,
                "missing": 1.0,
                "damaged": 1.2,
                "non_compliant": 1.5,
                "placement_error": 1.3
            }
            
            impact_multiplier = {
                "faded": 0.8,
                "missing": 1.0,
                "damaged": 0.9,
                "non_compliant": 1.1,
                "placement_error": 0.7
            }
            
            intervention = {
                "problem_type": prob_type,
                "category": "road_marking",
                "intervention_name": f"{marking_name} - {prob_type.replace('_', ' ').title()}",
                "description": f"{description} addressing {prob_type.replace('_', ' ')} issue",
                "cost_estimate": {
                    "materials": int(base_cost * cost_multiplier[prob_type] * 0.7),
                    "labor": int(base_cost * cost_multiplier[prob_type] * 0.3),
                    "total": int(base_cost * cost_multiplier[prob_type]),
                    "currency": "INR"
                },
                "predicted_impact": {
                    "accident_reduction_percent": int(base_impact * impact_multiplier[prob_type]),
                    "confidence_level": "high" if prob_type in ["missing", "non_compliant"] else "medium",
                    "lives_saved_per_year": round(base_impact * impact_multiplier[prob_type] / 15, 1),
                    "injury_prevention_per_year": round(base_impact * impact_multiplier[prob_type] / 4, 1)
                },
                "implementation_timeline": 1 if prob_type == "faded" else 3,
                "references": [
                    {
                        "standard": "IRC35-2015",
                        "clause": "8.2",
                        "page": 89,
                        "description": f"{marking_name} specifications and application requirements"
                    }
                ],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "prerequisites": {
                    "surface_preparation": True,
                    "traffic_management": True,
                    "weather_conditions": True
                },
                "compliance_requirements": [
                    "IRC 35-2015 compliance",
                    "MoRTH specifications",
                    "Local authority approval"
                ],
                "maintenance_requirements": {
                    "inspection_frequency": "monthly",
                    "repainting_cycle": "2_years",
                    "cleaning_frequency": "weekly"
                }
            }
            
            variations.append(intervention)
        
        return variations
    
    def _generate_traffic_calming_interventions(self) -> List[Dict[str, Any]]:
        """Generate traffic calming interventions"""
        interventions = []
        
        calming_types = [
            ("Speed Hump", "speed_hump", 25000, 40, "Install speed hump with proper signage"),
            ("Speed Table", "speed_table", 35000, 45, "Install speed table for pedestrian safety"),
            ("Rumble Strips", "rumble_strips", 15000, 30, "Install rumble strips for speed reduction"),
            ("Chicane", "chicane", 50000, 50, "Install chicane for speed control"),
            ("Raised Crosswalk", "raised_crosswalk", 40000, 55, "Install raised crosswalk"),
            ("Traffic Circle", "traffic_circle", 100000, 60, "Install mini traffic circle"),
            ("Gateway Treatment", "gateway_treatment", 30000, 35, "Install gateway treatment"),
            ("Lateral Shift", "lateral_shift", 20000, 25, "Install lateral shift"),
            ("Pinch Point", "pinch_point", 25000, 30, "Install pinch point"),
            ("Curb Extension", "curb_extension", 35000, 40, "Install curb extension"),
            ("Pedestrian Refuge", "pedestrian_refuge", 30000, 45, "Install pedestrian refuge island"),
            ("Bus Boarding Island", "bus_island", 40000, 35, "Install bus boarding island"),
            ("Cycle Track", "cycle_track", 80000, 50, "Install dedicated cycle track"),
            ("Shared Use Path", "shared_path", 60000, 40, "Install shared use path"),
            ("Traffic Signal", "traffic_signal", 200000, 65, "Install traffic signal with pedestrian phase")
        ]
        
        for calming_name, problem_type, base_cost, base_impact, description in calming_types:
            interventions.extend(self._create_calming_variations(calming_name, problem_type, base_cost, base_impact, description))
        
        return interventions
    
    def _create_calming_variations(self, calming_name: str, problem_type: str, base_cost: int, base_impact: int, description: str) -> List[Dict[str, Any]]:
        """Create variations of traffic calming interventions"""
        variations = []
        
        problem_types = ["missing", "damaged", "ineffective", "non_compliant", "maintenance_required"]
        
        for prob_type in problem_types:
            cost_multiplier = {
                "missing": 1.0,
                "damaged": 1.3,
                "ineffective": 1.5,
                "non_compliant": 1.8,
                "maintenance_required": 0.6
            }
            
            impact_multiplier = {
                "missing": 1.0,
                "damaged": 0.8,
                "ineffective": 0.7,
                "non_compliant": 1.1,
                "maintenance_required": 0.9
            }
            
            intervention = {
                "problem_type": prob_type,
                "category": "traffic_calming",
                "intervention_name": f"{calming_name} - {prob_type.replace('_', ' ').title()}",
                "description": f"{description} addressing {prob_type.replace('_', ' ')} issue",
                "cost_estimate": {
                    "materials": int(base_cost * cost_multiplier[prob_type] * 0.6),
                    "labor": int(base_cost * cost_multiplier[prob_type] * 0.4),
                    "total": int(base_cost * cost_multiplier[prob_type]),
                    "currency": "INR"
                },
                "predicted_impact": {
                    "accident_reduction_percent": int(base_impact * impact_multiplier[prob_type]),
                    "confidence_level": "high" if prob_type in ["missing", "non_compliant"] else "medium",
                    "lives_saved_per_year": round(base_impact * impact_multiplier[prob_type] / 10, 1),
                    "injury_prevention_per_year": round(base_impact * impact_multiplier[prob_type] / 3, 1)
                },
                "implementation_timeline": 7 if prob_type == "maintenance_required" else 14,
                "references": [
                    {
                        "standard": "IRC103-2012",
                        "clause": "6.3",
                        "page": 78,
                        "description": f"{calming_name} design and installation guidelines"
                    }
                ],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "prerequisites": {
                    "traffic_study": True,
                    "community_consultation": True,
                    "engineering_design": True,
                    "environmental_clearance": prob_type == "missing"
                },
                "compliance_requirements": [
                    "IRC 103-2012 compliance",
                    "MoRTH approval",
                    "Local authority permission",
                    "Environmental clearance"
                ],
                "maintenance_requirements": {
                    "inspection_frequency": "monthly",
                    "repair_cycle": "2_years",
                    "cleaning_frequency": "weekly"
                }
            }
            
            variations.append(intervention)
        
        return variations
    
    def _generate_infrastructure_interventions(self) -> List[Dict[str, Any]]:
        """Generate infrastructure interventions"""
        interventions = []
        
        infrastructure_types = [
            ("Street Lighting", "street_lighting", 50000, 35, "Install LED street lighting"),
            ("Guard Rail", "guard_rail", 30000, 40, "Install guard rail for safety"),
            ("Crash Barrier", "crash_barrier", 80000, 50, "Install crash barrier"),
            ("Median Barrier", "median_barrier", 60000, 45, "Install median barrier"),
            ("Pedestrian Bridge", "pedestrian_bridge", 500000, 70, "Install pedestrian bridge"),
            ("Underpass", "underpass", 800000, 75, "Install pedestrian underpass"),
            ("Footpath", "footpath", 100000, 60, "Construct footpath"),
            ("Drainage System", "drainage", 150000, 30, "Install drainage system"),
            ("Road Widening", "road_widening", 200000, 40, "Widen road section"),
            ("Intersection Improvement", "intersection", 300000, 55, "Improve intersection design"),
            ("Bus Shelter", "bus_shelter", 40000, 25, "Install bus shelter"),
            ("Traffic Island", "traffic_island", 25000, 35, "Install traffic island"),
            ("Retaining Wall", "retaining_wall", 120000, 30, "Construct retaining wall"),
            ("Road Surface", "road_surface", 180000, 25, "Resurface road"),
            ("Drainage Culvert", "culvert", 80000, 20, "Install drainage culvert")
        ]
        
        for infra_name, problem_type, base_cost, base_impact, description in infrastructure_types:
            interventions.extend(self._create_infrastructure_variations(infra_name, problem_type, base_cost, base_impact, description))
        
        return interventions
    
    def _create_infrastructure_variations(self, infra_name: str, problem_type: str, base_cost: int, base_impact: int, description: str) -> List[Dict[str, Any]]:
        """Create variations of infrastructure interventions"""
        variations = []
        
        problem_types = ["missing", "damaged", "inadequate", "non_compliant", "maintenance_required"]
        
        for prob_type in problem_types:
            cost_multiplier = {
                "missing": 1.0,
                "damaged": 1.4,
                "inadequate": 1.6,
                "non_compliant": 1.8,
                "maintenance_required": 0.5
            }
            
            impact_multiplier = {
                "missing": 1.0,
                "damaged": 0.7,
                "inadequate": 0.8,
                "non_compliant": 1.2,
                "maintenance_required": 0.6
            }
            
            intervention = {
                "problem_type": prob_type,
                "category": "infrastructure",
                "intervention_name": f"{infra_name} - {prob_type.replace('_', ' ').title()}",
                "description": f"{description} addressing {prob_type.replace('_', ' ')} issue",
                "cost_estimate": {
                    "materials": int(base_cost * cost_multiplier[prob_type] * 0.7),
                    "labor": int(base_cost * cost_multiplier[prob_type] * 0.3),
                    "total": int(base_cost * cost_multiplier[prob_type]),
                    "currency": "INR"
                },
                "predicted_impact": {
                    "accident_reduction_percent": int(base_impact * impact_multiplier[prob_type]),
                    "confidence_level": "high" if prob_type in ["missing", "non_compliant"] else "medium",
                    "lives_saved_per_year": round(base_impact * impact_multiplier[prob_type] / 8, 1),
                    "injury_prevention_per_year": round(base_impact * impact_multiplier[prob_type] / 2.5, 1)
                },
                "implementation_timeline": 30 if prob_type == "maintenance_required" else 60,
                "references": [
                    {
                        "standard": "IRC73-2018",
                        "clause": "5.2",
                        "page": 112,
                        "description": f"{infra_name} design and construction standards"
                    }
                ],
                "dependencies": [],
                "conflicts": [],
                "synergies": [],
                "prerequisites": {
                    "site_survey": True,
                    "engineering_design": True,
                    "environmental_clearance": prob_type == "missing",
                    "utility_coordination": True
                },
                "compliance_requirements": [
                    "IRC 73-2018 compliance",
                    "MoRTH approval",
                    "Local authority permission",
                    "Environmental clearance",
                    "Utility coordination"
                ],
                "maintenance_requirements": {
                    "inspection_frequency": "quarterly",
                    "repair_cycle": "5_years",
                    "cleaning_frequency": "monthly"
                }
            }
            
            variations.append(intervention)
        
        return variations

if __name__ == "__main__":
    # Generate comprehensive database
    db = InterventionDatabase()
    count = db.generate_comprehensive_database()
    db.save_interventions()
    print(f"Generated {count} interventions successfully!")
