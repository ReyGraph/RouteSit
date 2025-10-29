#!/usr/bin/env python3
"""
Accident Data Expander
Expands accident data collection to 10k-100k records for comprehensive training
"""

import json
import random
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class AccidentDataExpander:
    """Expands accident data with comprehensive road safety records"""
    
    def __init__(self):
        self.accident_records = []
        
        # Indian states and cities for realistic data
        self.locations = {
            "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
            "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"],
            "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirapalli"],
            "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
            "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
            "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Meerut"],
            "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri"],
            "Andhra Pradesh": ["Hyderabad", "Visakhapatnam", "Vijayawada", "Guntur", "Nellore"],
            "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
            "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar", "Patiala"]
        }
        
        # Road types and their characteristics
        self.road_types = {
            "highway": {"speed_limit": (80, 120), "traffic_volume": "high", "accident_rate": 0.15},
            "urban": {"speed_limit": (30, 60), "traffic_volume": "high", "accident_rate": 0.25},
            "rural": {"speed_limit": (40, 80), "traffic_volume": "medium", "accident_rate": 0.20},
            "city_center": {"speed_limit": (20, 40), "traffic_volume": "very_high", "accident_rate": 0.30},
            "residential": {"speed_limit": (20, 30), "traffic_volume": "low", "accident_rate": 0.10}
        }
        
        # Accident types and their characteristics
        self.accident_types = {
            "collision": {"severity_factor": 0.8, "common_causes": ["speeding", "distraction", "overtaking"]},
            "pedestrian_hit": {"severity_factor": 0.9, "common_causes": ["speeding", "poor_visibility", "jaywalking"]},
            "vehicle_overturn": {"severity_factor": 0.7, "common_causes": ["speeding", "poor_road_condition", "overloading"]},
            "head_on_collision": {"severity_factor": 0.95, "common_causes": ["wrong_lane", "overtaking", "fatigue"]},
            "rear_end": {"severity_factor": 0.4, "common_causes": ["tailgating", "sudden_braking", "distraction"]},
            "side_impact": {"severity_factor": 0.6, "common_causes": ["signal_violation", "speeding", "poor_judgment"]},
            "single_vehicle": {"severity_factor": 0.5, "common_causes": ["speeding", "fatigue", "mechanical_failure"]},
            "cyclist_hit": {"severity_factor": 0.85, "common_causes": ["speeding", "poor_visibility", "lane_violation"]}
        }
        
        # Severity levels
        self.severity_levels = ["fatal", "serious_injury", "minor_injury", "property_damage"]
        
        # Weather conditions
        self.weather_conditions = ["clear", "rain", "fog", "dust_storm", "cloudy", "haze"]
        
        # Time patterns (more accidents during certain hours)
        self.time_patterns = {
            "morning_rush": (7, 9, 0.25),
            "evening_rush": (17, 19, 0.30),
            "night": (22, 6, 0.15),
            "day": (9, 17, 0.20),
            "late_night": (0, 5, 0.10)
        }
        
        # Common interventions present/missing
        self.interventions = {
            "present": [
                "zebra_crossing", "speed_limit_sign", "traffic_signal", "speed_bump",
                "guard_rail", "street_lighting", "warning_sign", "stop_sign",
                "pedestrian_bridge", "traffic_calming", "median_barrier", "reflective_marking"
            ],
            "missing": [
                "advance_warning_sign", "flashing_beacon", "pedestrian_refuge",
                "tactile_paving", "audible_signal", "speed_camera", "rumble_strip",
                "chicane", "traffic_circle", "raised_crossing", "bike_lane"
            ]
        }
        
        # Vehicle types
        self.vehicle_types = [
            "car", "motorcycle", "truck", "bus", "auto_rickshaw", "bicycle",
            "pedestrian", "commercial_vehicle", "emergency_vehicle", "construction_vehicle"
        ]
    
    def generate_accident_record(self, accident_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Generate a single accident record"""
        
        # Select random location
        state = random.choice(list(self.locations.keys()))
        city = random.choice(self.locations[state])
        
        # Generate coordinates (approximate for Indian cities)
        lat = random.uniform(8.0, 37.0)  # India's latitude range
        lon = random.uniform(68.0, 97.0)  # India's longitude range
        
        # Select road type
        road_type = random.choice(list(self.road_types.keys()))
        road_chars = self.road_types[road_type]
        
        # Select accident type
        accident_type = random.choice(list(self.accident_types.keys()))
        accident_chars = self.accident_types[accident_type]
        
        # Determine severity based on accident type and road characteristics
        severity_prob = accident_chars["severity_factor"] * road_chars["accident_rate"]
        if severity_prob > 0.8:
            severity = "fatal"
        elif severity_prob > 0.6:
            severity = "serious_injury"
        elif severity_prob > 0.3:
            severity = "minor_injury"
        else:
            severity = "property_damage"
        
        # Select weather condition
        weather = random.choice(self.weather_conditions)
        
        # Determine time pattern
        hour = timestamp.hour
        time_pattern = "day"  # default
        for pattern, (start_hour, end_hour, _) in self.time_patterns.items():
            if start_hour <= hour < end_hour or (start_hour > end_hour and (hour >= start_hour or hour < end_hour)):
                time_pattern = pattern
                break
        
        # Generate interventions present/missing
        interventions_present = random.sample(
            self.interventions["present"], 
            random.randint(1, min(4, len(self.interventions["present"])))
        )
        
        interventions_missing = random.sample(
            self.interventions["missing"], 
            random.randint(1, min(3, len(self.interventions["missing"])))
        )
        
        # Generate vehicles involved
        vehicles_involved = random.sample(
            self.vehicle_types, 
            random.randint(1, min(3, len(self.vehicle_types)))
        )
        
        # Generate cost estimates
        cost_multipliers = {
            "fatal": (500000, 2000000),
            "serious_injury": (100000, 500000),
            "minor_injury": (10000, 100000),
            "property_damage": (5000, 50000)
        }
        
        cost_range = cost_multipliers[severity]
        estimated_cost = random.randint(cost_range[0], cost_range[1])
        
        # Generate impact assessment
        impact_assessment = {
            "traffic_disruption_hours": random.randint(1, 12),
            "emergency_response_time": random.randint(5, 45),
            "road_closure_duration": random.randint(0, 8),
            "economic_impact": estimated_cost * random.uniform(1.5, 3.0)
        }
        
        # Generate contributing factors
        contributing_factors = random.sample(
            accident_chars["common_causes"] + ["poor_road_condition", "weather", "vehicle_condition"],
            random.randint(1, 4)
        )
        
        return {
            "accident_id": accident_id,
            "timestamp": timestamp.isoformat(),
            "location": {
                "state": state,
                "city": city,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "address": f"{random.randint(1, 999)} Main Road, {city}, {state}",
                "road_type": road_type,
                "speed_limit": random.randint(road_chars["speed_limit"][0], road_chars["speed_limit"][1])
            },
            "accident_details": {
                "type": accident_type,
                "severity": severity,
                "vehicles_involved": vehicles_involved,
                "injuries": {
                    "fatal": 1 if severity == "fatal" else 0,
                    "serious": random.randint(0, 3) if severity in ["fatal", "serious_injury"] else 0,
                    "minor": random.randint(0, 5) if severity != "property_damage" else 0
                },
                "property_damage": estimated_cost,
                "weather_condition": weather,
                "time_of_day": hour,
                "day_of_week": timestamp.weekday(),
                "time_pattern": time_pattern
            },
            "interventions": {
                "present": interventions_present,
                "missing": interventions_missing,
                "effectiveness_score": random.uniform(0.3, 0.9)
            },
            "contributing_factors": contributing_factors,
            "impact_assessment": impact_assessment,
            "verification": {
                "verified": random.choice([True, False]),
                "source": random.choice(["police_report", "hospital_record", "insurance_claim", "witness_report"]),
                "confidence_score": random.uniform(0.6, 1.0)
            },
            "follow_up": {
                "investigation_status": random.choice(["completed", "ongoing", "pending"]),
                "legal_action": random.choice(["none", "filed", "settled", "pending"]),
                "preventive_measures": random.sample(self.interventions["missing"], random.randint(0, 3))
            }
        }
    
    def expand_accident_data(self, target_count: int = 50000):
        """Expand accident data to target count"""
        logger.info(f"Expanding accident data to {target_count} records...")
        
        # Start date for generating historical data
        start_date = datetime.now() - timedelta(days=365 * 3)  # 3 years of data
        
        # Generate accidents with realistic temporal distribution
        current_count = 0
        current_date = start_date
        
        while current_count < target_count:
            # Generate accidents for current day
            daily_accidents = random.randint(1, 5)  # 1-5 accidents per day
            
            for _ in range(daily_accidents):
                if current_count >= target_count:
                    break
                
                # Generate random time within the day
                random_hour = random.randint(0, 23)
                random_minute = random.randint(0, 59)
                accident_time = current_date.replace(hour=random_hour, minute=random_minute)
                
                # Generate accident ID
                accident_id = f"acc_{current_count + 1:08d}"
                
                # Generate accident record
                accident_record = self.generate_accident_record(accident_id, accident_time)
                self.accident_records.append(accident_record)
                
                current_count += 1
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Log progress
            if current_count % 5000 == 0:
                logger.info(f"Generated {current_count} accident records...")
        
        logger.info(f"Accident data expansion complete: {len(self.accident_records)} records")
    
    def save_accident_data(self):
        """Save expanded accident data"""
        try:
            # Save main database
            db_path = Path("data/accident_data/accident_records.json")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(self.accident_records, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved accident data to {db_path}")
            
            # Generate statistics
            self._generate_statistics()
            
            # Save sample data for quick access
            sample_data = self.accident_records[:1000]  # First 1000 records
            sample_path = Path("data/accident_data/sample_accidents.json")
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved sample data to {sample_path}")
            
        except Exception as e:
            logger.error(f"Error saving accident data: {e}")
            raise
    
    def _generate_statistics(self):
        """Generate accident data statistics"""
        stats = {
            "total_records": len(self.accident_records),
            "severity_distribution": {},
            "accident_type_distribution": {},
            "road_type_distribution": {},
            "state_distribution": {},
            "time_pattern_distribution": {},
            "cost_ranges": {
                "min": min(r["accident_details"]["property_damage"] for r in self.accident_records),
                "max": max(r["accident_details"]["property_damage"] for r in self.accident_records),
                "average": sum(r["accident_details"]["property_damage"] for r in self.accident_records) / len(self.accident_records)
            },
            "date_range": {
                "start": min(r["timestamp"] for r in self.accident_records),
                "end": max(r["timestamp"] for r in self.accident_records)
            }
        }
        
        # Count distributions
        for record in self.accident_records:
            severity = record["accident_details"]["severity"]
            stats["severity_distribution"][severity] = stats["severity_distribution"].get(severity, 0) + 1
            
            accident_type = record["accident_details"]["type"]
            stats["accident_type_distribution"][accident_type] = stats["accident_type_distribution"].get(accident_type, 0) + 1
            
            road_type = record["location"]["road_type"]
            stats["road_type_distribution"][road_type] = stats["road_type_distribution"].get(road_type, 0) + 1
            
            state = record["location"]["state"]
            stats["state_distribution"][state] = stats["state_distribution"].get(state, 0) + 1
            
            time_pattern = record["accident_details"]["time_pattern"]
            stats["time_pattern_distribution"][time_pattern] = stats["time_pattern_distribution"].get(time_pattern, 0) + 1
        
        # Save statistics
        stats_path = Path("data/accident_data/accident_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Accident statistics saved to {stats_path}")
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Severity distribution: {stats['severity_distribution']}")
        logger.info(f"Cost range: ₹{stats['cost_ranges']['min']:,} - ₹{stats['cost_ranges']['max']:,}")

async def main():
    """Main function to expand accident data"""
    logging.basicConfig(level=logging.INFO)
    
    expander = AccidentDataExpander()
    
    # Expand to 50,000 accident records
    expander.expand_accident_data(target_count=50000)
    
    # Save expanded data
    expander.save_accident_data()
    
    print("Accident data expansion completed successfully!")
    print(f"Total accident records: {len(expander.accident_records)}")
    print("Comprehensive accident database ready for ML training")

if __name__ == "__main__":
    asyncio.run(main())
