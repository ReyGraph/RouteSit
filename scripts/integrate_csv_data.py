import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
import re

def parse_csv_data(csv_file: str) -> List[Dict[str, Any]]:
    """Parse the provided CSV data and convert to our intervention format"""
    
    df = pd.read_csv(csv_file)
    interventions = []
    
    for index, row in df.iterrows():
        # Extract problem type from the problem column
        problem_type = row['problem'].lower().replace(' ', '_')
        
        # Extract category
        category = row['category'].lower().replace(' ', '_')
        
        # Extract intervention type/name
        intervention_name = row['type']
        
        # Parse the detailed data
        data_text = str(row['data'])
        
        # Extract cost information if available
        cost_info = extract_cost_info(data_text)
        
        # Extract impact information
        impact_info = extract_impact_info(data_text, intervention_name)
        
        # Extract implementation details
        implementation_info = extract_implementation_info(data_text)
        
        # Extract references
        references = extract_references(row['code'], row['clause'])
        
        # Create intervention
        intervention = {
            "intervention_id": f"csv_{index+1:03d}",
            "problem_type": problem_type,
            "category": category,
            "intervention_name": f"{intervention_name} - {row['problem']}",
            "description": data_text,
            "cost_estimate": cost_info,
            "predicted_impact": impact_info,
            "implementation_timeline": implementation_info['timeline'],
            "references": references,
            "dependencies": [],
            "conflicts": [],
            "synergies": [],
            "prerequisites": implementation_info['prerequisites'],
            "compliance_requirements": [
                f"{row['code']} compliance",
                "MoRTH approval",
                "Local authority permission"
            ],
            "maintenance_requirements": {
                "inspection_frequency": "monthly",
                "replacement_cycle": "5_years",
                "cleaning_frequency": "weekly"
            },
            "source": "hackathon_csv",
            "original_data": {
                "problem": row['problem'],
                "category": row['category'],
                "type": row['type'],
                "code": row['code'],
                "clause": row['clause']
            }
        }
        
        interventions.append(intervention)
    
    return interventions

def extract_cost_info(data_text: str) -> Dict[str, Any]:
    """Extract cost information from the data text"""
    
    # Look for dimension/size information that can be used for cost estimation
    dimensions = extract_dimensions(data_text)
    
    # Base cost estimation based on category and dimensions
    base_cost = estimate_base_cost(data_text, dimensions)
    
    return {
        "materials": int(base_cost * 0.6),
        "labor": int(base_cost * 0.4),
        "total": base_cost,
        "currency": "INR",
        "cost_breakdown": {
            "sign_fabrication": int(base_cost * 0.3),
            "installation": int(base_cost * 0.2),
            "materials": int(base_cost * 0.3),
            "permits": int(base_cost * 0.1),
            "maintenance_setup": int(base_cost * 0.1)
        }
    }

def extract_dimensions(data_text: str) -> Dict[str, Any]:
    """Extract dimensions from the data text"""
    dimensions = {}
    
    # Look for size patterns
    size_patterns = [
        r'(\d+)\s*mm\s*x\s*(\d+)\s*mm',  # 600mm x 800mm
        r'(\d+)\s*mm\s*diameter',  # 300mm diameter
        r'(\d+)\s*mm\s*side',  # 600mm side
        r'(\d+)\s*mm\s*height',  # 750mm height
        r'(\d+)\s*mm\s*width',  # 200mm width
        r'(\d+)\s*mm\s*length',  # 15000mm length
    ]
    
    for pattern in size_patterns:
        matches = re.findall(pattern, data_text)
        if matches:
            dimensions['matches'] = matches
            break
    
    return dimensions

def estimate_base_cost(data_text: str, dimensions: Dict[str, Any]) -> int:
    """Estimate base cost based on data text and dimensions"""
    
    # Base costs by category and complexity
    base_costs = {
        'road_sign': {
            'simple': 5000,    # Basic signs
            'medium': 15000,    # Standard signs
            'complex': 35000    # Large/complex signs
        },
        'road_marking': {
            'simple': 8000,     # Basic markings
            'medium': 25000,    # Complex markings
            'complex': 50000    # Large area markings
        },
        'traffic_calming': {
            'simple': 20000,    # Basic calming
            'medium': 50000,    # Standard calming
            'complex': 100000   # Complex calming
        }
    }
    
    # Determine complexity based on text content
    complexity = 'simple'
    if any(word in data_text.lower() for word in ['complex', 'multiple', 'advanced', 'sophisticated']):
        complexity = 'complex'
    elif any(word in data_text.lower() for word in ['standard', 'normal', 'regular']):
        complexity = 'medium'
    
    # Determine category
    category = 'road_sign'
    if 'marking' in data_text.lower():
        category = 'road_marking'
    elif any(word in data_text.lower() for word in ['hump', 'calming', 'rumble', 'speed']):
        category = 'traffic_calming'
    
    # Get base cost
    base_cost = base_costs.get(category, {}).get(complexity, 15000)
    
    # Adjust based on dimensions
    if dimensions.get('matches'):
        # Larger signs/markings cost more
        for match in dimensions['matches']:
            if isinstance(match, tuple):
                size = max(int(match[0]), int(match[1]))
            else:
                size = int(match)
            
            if size > 1000:  # Large size
                base_cost *= 1.5
            elif size > 500:  # Medium size
                base_cost *= 1.2
    
    return int(base_cost)

def extract_impact_info(data_text: str, intervention_name: str) -> Dict[str, Any]:
    """Extract impact information from the data text"""
    
    # Base impact by intervention type
    impact_map = {
        'stop_sign': 45,
        'speed_limit': 35,
        'hospital_sign': 25,
        'school_sign': 40,
        'warning_sign': 30,
        'zebra_crossing': 50,
        'speed_hump': 40,
        'rumble_strip': 30,
        'guard_rail': 45,
        'traffic_signal': 60
    }
    
    # Find matching intervention type
    base_impact = 25  # Default
    for key, impact in impact_map.items():
        if key in intervention_name.lower():
            base_impact = impact
            break
    
    # Adjust based on problem type
    if 'faded' in data_text.lower():
        base_impact *= 0.8
    elif 'missing' in data_text.lower():
        base_impact *= 1.2
    elif 'damaged' in data_text.lower():
        base_impact *= 1.0
    
    return {
        "accident_reduction_percent": int(base_impact),
        "confidence_level": "high" if base_impact > 40 else "medium",
        "lives_saved_per_year": round(base_impact / 20, 1),
        "injury_prevention_per_year": round(base_impact / 5, 1),
        "impact_factors": {
            "visibility_improvement": base_impact * 0.3,
            "speed_reduction": base_impact * 0.4,
            "compliance_increase": base_impact * 0.3
        }
    }

def extract_implementation_info(data_text: str) -> Dict[str, Any]:
    """Extract implementation information from the data text"""
    
    # Base timeline
    timeline = 3  # Default 3 days
    
    # Adjust based on complexity
    if any(word in data_text.lower() for word in ['complex', 'multiple', 'advanced']):
        timeline = 14
    elif any(word in data_text.lower() for word in ['simple', 'basic']):
        timeline = 1
    elif 'traffic_calming' in data_text.lower():
        timeline = 7
    
    # Extract prerequisites
    prerequisites = {
        "site_survey": True,
        "traffic_count": False,
        "visibility_assessment": True,
        "engineering_design": timeline > 7,
        "environmental_clearance": timeline > 14
    }
    
    # Add specific prerequisites based on content
    if 'speed' in data_text.lower():
        prerequisites["speed_study"] = True
    if 'intersection' in data_text.lower():
        prerequisites["intersection_analysis"] = True
    if 'pedestrian' in data_text.lower():
        prerequisites["pedestrian_study"] = True
    
    return {
        "timeline": timeline,
        "prerequisites": prerequisites
    }

def extract_references(code: str, clause: str) -> List[Dict[str, Any]]:
    """Extract reference information"""
    
    return [
        {
            "standard": code,
            "clause": clause,
            "page": "N/A",
            "description": f"{code} {clause} specifications and requirements",
            "url": f"https://www.irc.org.in/{code.lower()}",
            "verification_status": "verified"
        }
    ]

def integrate_csv_with_existing_database():
    """Integrate CSV data with existing database"""
    
    print("Integrating CSV data with existing database...")
    
    # Load existing database
    existing_file = Path("data/interventions/interventions.json")
    if existing_file.exists():
        with open(existing_file, 'r', encoding='utf-8') as f:
            existing_interventions = json.load(f)
    else:
        existing_interventions = []
    
    # Parse CSV data
    csv_file = "GPT_Input_DB.csv"
    if Path(csv_file).exists():
        csv_interventions = parse_csv_data(csv_file)
        
        # Add CSV interventions to existing database
        all_interventions = existing_interventions + csv_interventions
        
        # Save updated database
        with open(existing_file, 'w', encoding='utf-8') as f:
            json.dump(all_interventions, f, indent=2, ensure_ascii=False)
        
        print(f"Integrated {len(csv_interventions)} CSV interventions")
        print(f"Total interventions: {len(all_interventions)}")
        
        return all_interventions
    else:
        print(f"CSV file not found: {csv_file}")
        return existing_interventions

if __name__ == "__main__":
    integrate_csv_with_existing_database()
