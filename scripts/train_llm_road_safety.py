#!/usr/bin/env python3
"""
LLM Training Data Generator for Road Safety Domain
Creates comprehensive training dataset for fine-tuning Llama 3 8B on road safety scenarios
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import random
import re

logger = logging.getLogger(__name__)

class RoadSafetyTrainingDataGenerator:
    """Generate comprehensive training data for road safety LLM fine-tuning"""
    
    def __init__(self):
        self.output_dir = Path("data/training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self.interventions_db = self._load_interventions()
        self.accident_data = self._load_accident_data()
        
        # Training data templates
        self.templates = {
            "intervention_recommendation": {
                "instruction": "Analyze this road safety situation and recommend appropriate interventions with detailed reasoning.",
                "input_template": "Road Safety Situation: {description}\nLocation: {location}\nRoad Type: {road_type}\nTraffic Volume: {traffic_volume}\nWeather: {weather}\nInterventions Present: {interventions_present}\nInterventions Missing: {interventions_missing}",
                "output_template": "Based on the analysis, I recommend the following interventions:\n\n1. **Primary Intervention**: {primary_intervention}\n   - Cost: ₹{cost:,}\n   - Timeline: {timeline} days\n   - Expected Impact: {impact}% accident reduction\n   - Reasoning: {reasoning}\n\n2. **Supporting Interventions**: {supporting_interventions}\n\n3. **Cascading Effects**: {cascading_effects}\n\n4. **Implementation Priority**: {priority}\n\n5. **Compliance**: {compliance}\n\n6. **References**: {references}"
            },
            
            "cascading_effects": {
                "instruction": "Predict the cascading effects and secondary impacts of implementing this road safety intervention.",
                "input_template": "Intervention: {intervention}\nLocation: {location}\nRoad Context: {road_context}\nTraffic Patterns: {traffic_patterns}",
                "output_template": "Cascading Effects Analysis:\n\n**Primary Effects**:\n{primary_effects}\n\n**Secondary Effects**:\n{secondary_effects}\n\n**Potential Conflicts**:\n{conflicts}\n\n**Synergistic Opportunities**:\n{synergies}\n\n**Risk Assessment**:\n{risk_assessment}\n\n**Mitigation Strategies**:\n{mitigation}"
            },
            
            "cost_benefit_analysis": {
                "instruction": "Perform a comprehensive cost-benefit analysis for this road safety intervention.",
                "input_template": "Intervention: {intervention}\nLocation: {location}\nBudget: ₹{budget:,}\nTimeline: {timeline} days\nExpected Traffic: {traffic_volume}",
                "output_template": "Cost-Benefit Analysis:\n\n**Cost Breakdown**:\n- Materials: ₹{materials:,}\n- Labor: ₹{labor:,}\n- Permits: ₹{permits:,}\n- Total Cost: ₹{total_cost:,}\n\n**Benefits Quantification**:\n- Lives Saved (Annual): {lives_saved}\n- Injuries Prevented (Annual): {injuries_prevented}\n- Property Damage Avoided: ₹{property_damage:,}\n- Economic Value: ₹{economic_value:,}\n\n**ROI Analysis**:\n- Payback Period: {payback_period} years\n- Net Present Value: ₹{npv:,}\n- Benefit-Cost Ratio: {bcr}\n\n**Sensitivity Analysis**:\n{sensitivity_analysis}\n\n**Recommendation**: {recommendation}"
            },
            
            "implementation_planning": {
                "instruction": "Create a detailed implementation plan for this road safety intervention.",
                "input_template": "Intervention: {intervention}\nLocation: {location}\nBudget: ₹{budget:,}\nTimeline: {timeline} days\nStakeholders: {stakeholders}",
                "output_template": "Implementation Plan:\n\n**Phase 1: Pre-Implementation (Days 1-{phase1_days})**\n{tasks_phase1}\n\n**Phase 2: Implementation (Days {phase2_start}-{phase2_end})**\n{tasks_phase2}\n\n**Phase 3: Post-Implementation (Days {phase3_start}-{phase3_end})**\n{tasks_phase3}\n\n**Resource Requirements**:\n- Materials: {materials_list}\n- Labor: {labor_requirements}\n- Equipment: {equipment_list}\n\n**Quality Control**:\n{qc_checkpoints}\n\n**Compliance Verification**:\n{compliance_checks}\n\n**Risk Management**:\n{risk_mitigation}\n\n**Success Metrics**:\n{success_metrics}"
            },
            
            "accident_analysis": {
                "instruction": "Analyze this accident scenario and identify contributing factors and prevention strategies.",
                "input_template": "Accident Details:\nLocation: {location}\nDate: {date}\nTime: {time}\nType: {accident_type}\nSeverity: {severity}\nVehicles Involved: {vehicles}\nWeather: {weather}\nRoad Conditions: {road_conditions}\nInterventions Present: {interventions_present}",
                "output_template": "Accident Analysis:\n\n**Contributing Factors**:\n{contributing_factors}\n\n**Root Cause Analysis**:\n{root_causes}\n\n**Prevention Strategies**:\n{prevention_strategies}\n\n**Intervention Recommendations**:\n{intervention_recommendations}\n\n**Policy Implications**:\n{policy_implications}\n\n**Lessons Learned**:\n{lessons_learned}"
            }
        }
        
        # Road safety scenarios
        self.scenarios = self._generate_scenarios()
    
    def _load_interventions(self) -> List[Dict]:
        """Load intervention database"""
        try:
            with open("data/interventions/interventions_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load interventions: {e}")
            return []
    
    def _load_accident_data(self) -> List[Dict]:
        """Load accident data"""
        try:
            with open("data/accident_data/accident_records.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load accident data: {e}")
            return []
    
    def _generate_scenarios(self) -> List[Dict]:
        """Generate diverse road safety scenarios"""
        scenarios = [
            {
                "description": "Faded zebra crossing at school zone intersection with high pedestrian traffic during morning rush hour",
                "location": "Near ABC School, Mumbai",
                "road_type": "urban",
                "traffic_volume": "high",
                "weather": "clear",
                "interventions_present": ["school_zone_sign", "speed_limit_sign"],
                "interventions_missing": ["zebra_crossing", "pedestrian_refuge", "flashing_beacon"]
            },
            {
                "description": "Missing STOP sign at T-junction causing frequent near-misses and accidents",
                "location": "Highway intersection, Delhi",
                "road_type": "highway",
                "traffic_volume": "very_high",
                "weather": "clear",
                "interventions_present": ["lane_markings"],
                "interventions_missing": ["stop_sign", "advance_warning_sign", "rumble_strips"]
            },
            {
                "description": "Damaged guard rail on mountain road with sharp curves and steep drop-offs",
                "location": "Hill station road, Himachal Pradesh",
                "road_type": "rural",
                "traffic_volume": "medium",
                "weather": "foggy",
                "interventions_present": ["curve_warning_sign"],
                "interventions_missing": ["guard_rail", "reflective_markings", "speed_bumps"]
            },
            {
                "description": "Inadequate lighting on pedestrian crossing near hospital causing night-time accidents",
                "location": "Near City Hospital, Bangalore",
                "road_type": "urban",
                "traffic_volume": "high",
                "weather": "clear",
                "interventions_present": ["zebra_crossing", "pedestrian_sign"],
                "interventions_missing": ["street_lighting", "flashing_beacon", "tactile_paving"]
            },
            {
                "description": "Speed hump missing on residential street with speeding vehicles near playground",
                "location": "Residential area, Chennai",
                "road_type": "residential",
                "traffic_volume": "low",
                "weather": "clear",
                "interventions_present": ["playground_sign"],
                "interventions_missing": ["speed_hump", "speed_limit_sign", "traffic_calming"]
            }
        ]
        
        # Generate additional scenarios from accident data
        for accident in self.accident_data[:100]:  # Use first 100 accidents
            scenario = {
                "description": f"Accident scenario: {accident['accident_details']['type']} with {accident['accident_details']['severity']} severity",
                "location": f"{accident['location']['city']}, {accident['location']['state']}",
                "road_type": accident['location']['road_type'],
                "traffic_volume": accident['accident_details'].get('traffic_volume', 'medium'),
                "weather": accident['accident_details']['weather_condition'],
                "interventions_present": accident['interventions']['present'],
                "interventions_missing": accident['interventions']['missing']
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_intervention_recommendations(self, count: int = 2000) -> List[Dict]:
        """Generate intervention recommendation training examples"""
        examples = []
        
        for i in range(count):
            scenario = random.choice(self.scenarios)
            intervention = random.choice(self.interventions_db)
            
            # Generate realistic values
            base_cost = intervention['cost_estimate']['total']
            cost = random.randint(int(base_cost * 0.8), int(base_cost * 1.2))
            timeline = random.randint(1, 30)
            impact = random.randint(20, 80)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(scenario, intervention)
            
            # Generate supporting interventions
            supporting = random.sample(self.interventions_db, random.randint(1, 3))
            supporting_text = ", ".join([s['intervention_name'] for s in supporting])
            
            # Generate cascading effects
            cascading = self._generate_cascading_effects(intervention)
            
            # Generate compliance info
            compliance = f"Compliant with {random.choice(['IRC67-2022', 'IRC35-2015', 'MoRTH-2018'])} standards"
            
            # Generate references
            references = f"IRC67-2022 Clause {random.randint(10, 20)}.{random.randint(1, 9)}, MoRTH Guidelines 2018"
            
            example = {
                "instruction": self.templates["intervention_recommendation"]["instruction"],
                "input": self.templates["intervention_recommendation"]["input_template"].format(
                    description=scenario["description"],
                    location=scenario["location"],
                    road_type=scenario["road_type"],
                    traffic_volume=scenario["traffic_volume"],
                    weather=scenario["weather"],
                    interventions_present=", ".join(scenario["interventions_present"]),
                    interventions_missing=", ".join(scenario["interventions_missing"])
                ),
                "output": self.templates["intervention_recommendation"]["output_template"].format(
                    primary_intervention=intervention['intervention_name'],
                    cost=cost,
                    timeline=timeline,
                    impact=impact,
                    reasoning=reasoning,
                    supporting_interventions=supporting_text,
                    cascading_effects=cascading,
                    priority=random.choice(['High', 'Medium', 'Low']),
                    compliance=compliance,
                    references=references
                )
            }
            
            examples.append(example)
        
        return examples
    
    def generate_cascading_effects(self, count: int = 1500) -> List[Dict]:
        """Generate cascading effects training examples"""
        examples = []
        
        for i in range(count):
            intervention = random.choice(self.interventions_db)
            scenario = random.choice(self.scenarios)
            
            # Generate cascading effects
            primary_effects = self._generate_primary_effects(intervention)
            secondary_effects = self._generate_secondary_effects(intervention)
            conflicts = self._generate_conflicts(intervention)
            synergies = self._generate_synergies(intervention)
            risk_assessment = self._generate_risk_assessment(intervention)
            mitigation = self._generate_mitigation_strategies(intervention)
            
            example = {
                "instruction": self.templates["cascading_effects"]["instruction"],
                "input": self.templates["cascading_effects"]["input_template"].format(
                    intervention=intervention['intervention_name'],
                    location=scenario['location'],
                    road_context=f"{scenario['road_type']} road with {scenario['traffic_volume']} traffic",
                    traffic_patterns=f"Peak hours: {random.choice(['morning', 'evening'])} rush, Weather: {scenario['weather']}"
                ),
                "output": self.templates["cascading_effects"]["output_template"].format(
                    primary_effects=primary_effects,
                    secondary_effects=secondary_effects,
                    conflicts=conflicts,
                    synergies=synergies,
                    risk_assessment=risk_assessment,
                    mitigation=mitigation
                )
            }
            
            examples.append(example)
        
        return examples
    
    def generate_cost_benefit_analysis(self, count: int = 1500) -> List[Dict]:
        """Generate cost-benefit analysis training examples"""
        examples = []
        
        for i in range(count):
            intervention = random.choice(self.interventions_db)
            scenario = random.choice(self.scenarios)
            
            # Generate cost breakdown
            base_cost = intervention['cost_estimate']['total']
            total_cost = random.randint(int(base_cost * 0.8), int(base_cost * 1.2))
            materials = int(total_cost * random.uniform(0.4, 0.7))
            labor = int(total_cost * random.uniform(0.2, 0.4))
            permits = int(total_cost * random.uniform(0.05, 0.15))
            
            # Generate benefits
            lives_saved = random.uniform(0.5, 5.0)
            injuries_prevented = random.uniform(2.0, 15.0)
            property_damage = random.randint(50000, 500000)
            economic_value = int(lives_saved * 1000000 + injuries_prevented * 200000 + property_damage)
            
            # Calculate ROI metrics
            payback_period = random.uniform(1.0, 10.0)
            npv = economic_value - total_cost
            bcr = economic_value / total_cost
            
            example = {
                "instruction": self.templates["cost_benefit_analysis"]["instruction"],
                "input": self.templates["cost_benefit_analysis"]["input_template"].format(
                    intervention=intervention['intervention_name'],
                    location=scenario['location'],
                    budget=total_cost,
                    timeline=random.randint(1, 30),
                    traffic_volume=scenario['traffic_volume']
                ),
                "output": self.templates["cost_benefit_analysis"]["output_template"].format(
                    materials=materials,
                    labor=labor,
                    permits=permits,
                    total_cost=total_cost,
                    lives_saved=f"{lives_saved:.1f}",
                    injuries_prevented=f"{injuries_prevented:.1f}",
                    property_damage=property_damage,
                    economic_value=economic_value,
                    payback_period=f"{payback_period:.1f}",
                    npv=npv,
                    bcr=f"{bcr:.2f}",
                    sensitivity_analysis=self._generate_sensitivity_analysis(),
                    recommendation=self._generate_recommendation(bcr)
                )
            }
            
            examples.append(example)
        
        return examples
    
    def generate_implementation_plans(self, count: int = 1000) -> List[Dict]:
        """Generate implementation planning training examples"""
        examples = []
        
        for i in range(count):
            intervention = random.choice(self.interventions_db)
            scenario = random.choice(self.scenarios)
            
            # Generate implementation phases
            total_timeline = random.randint(5, 60)
            phase1_days = random.randint(1, 5)
            phase2_start = phase1_days + 1
            phase2_end = total_timeline - random.randint(1, 5)
            phase3_start = phase2_end + 1
            phase3_end = total_timeline
            
            example = {
                "instruction": self.templates["implementation_planning"]["instruction"],
                "input": self.templates["implementation_planning"]["input_template"].format(
                    intervention=intervention['intervention_name'],
                    location=scenario['location'],
                    budget=random.randint(10000, 500000),
                    timeline=total_timeline,
                    stakeholders="Traffic Police, Local Municipality, Contractors"
                ),
                "output": self.templates["implementation_planning"]["output_template"].format(
                    phase1_days=phase1_days,
                    tasks_phase1=self._generate_phase1_tasks(),
                    phase2_start=phase2_start,
                    phase2_end=phase2_end,
                    tasks_phase2=self._generate_phase2_tasks(),
                    phase3_start=phase3_start,
                    phase3_end=phase3_end,
                    tasks_phase3=self._generate_phase3_tasks(),
                    materials_list=self._generate_materials_list(intervention),
                    labor_requirements=self._generate_labor_requirements(),
                    equipment_list=self._generate_equipment_list(),
                    qc_checkpoints=self._generate_qc_checkpoints(),
                    compliance_checks=self._generate_compliance_checks(),
                    risk_mitigation=self._generate_risk_mitigation(),
                    success_metrics=self._generate_success_metrics()
                )
            }
            
            examples.append(example)
        
        return examples
    
    def generate_accident_analyses(self, count: int = 1000) -> List[Dict]:
        """Generate accident analysis training examples"""
        examples = []
        
        for i in range(count):
            accident = random.choice(self.accident_data)
            
            example = {
                "instruction": self.templates["accident_analysis"]["instruction"],
                "input": self.templates["accident_analysis"]["input_template"].format(
                    location=f"{accident['location']['city']}, {accident['location']['state']}",
                    date=accident['timestamp'][:10],
                    time=accident['timestamp'][11:16],
                    accident_type=accident['accident_details']['type'],
                    severity=accident['accident_details']['severity'],
                    vehicles=", ".join(accident['accident_details']['vehicles_involved']),
                    weather=accident['accident_details']['weather_condition'],
                    road_conditions=f"{accident['location']['road_type']} road",
                    interventions_present=", ".join(accident['interventions']['present'])
                ),
                "output": self.templates["accident_analysis"]["output_template"].format(
                    contributing_factors=self._generate_contributing_factors(accident),
                    root_causes=self._generate_root_causes(accident),
                    prevention_strategies=self._generate_prevention_strategies(accident),
                    intervention_recommendations=self._generate_intervention_recommendations(accident),
                    policy_implications=self._generate_policy_implications(accident),
                    lessons_learned=self._generate_lessons_learned(accident)
                )
            }
            
            examples.append(example)
        
        return examples
    
    def _generate_reasoning(self, scenario: Dict, intervention: Dict) -> str:
        """Generate reasoning for intervention recommendation"""
        reasons = [
            f"Addresses the {scenario['description'].split()[0]} issue identified",
            f"Suitable for {scenario['road_type']} road conditions",
            f"Appropriate for {scenario['traffic_volume']} traffic volume",
            f"Complements existing interventions: {', '.join(scenario['interventions_present'])}",
            f"Fills critical gap in safety infrastructure"
        ]
        return "; ".join(random.sample(reasons, 3))
    
    def _generate_cascading_effects(self, intervention: Dict) -> str:
        """Generate cascading effects"""
        effects = [
            "Improved traffic flow efficiency",
            "Reduced vehicle speeds in the area",
            "Enhanced pedestrian safety",
            "Increased driver awareness",
            "Reduced noise pollution",
            "Improved air quality"
        ]
        return "; ".join(random.sample(effects, 3))
    
    def _generate_primary_effects(self, intervention: Dict) -> str:
        """Generate primary effects"""
        effects = [
            f"Direct improvement in {intervention['intervention_name'].lower()} effectiveness",
            "Immediate safety enhancement",
            "Clear visibility improvement",
            "Traffic behavior modification"
        ]
        return "\n".join([f"- {effect}" for effect in effects])
    
    def _generate_secondary_effects(self, intervention: Dict) -> str:
        """Generate secondary effects"""
        effects = [
            "Reduced accident rates in surrounding areas",
            "Improved driver compliance with traffic rules",
            "Enhanced pedestrian confidence",
            "Positive impact on local business accessibility"
        ]
        return "\n".join([f"- {effect}" for effect in effects])
    
    def _generate_conflicts(self, intervention: Dict) -> str:
        """Generate potential conflicts"""
        conflicts = [
            "May require temporary traffic diversion",
            "Could impact emergency vehicle access",
            "May affect local business operations during installation",
            "Potential noise during construction phase"
        ]
        return "\n".join([f"- {conflict}" for conflict in conflicts])
    
    def _generate_synergies(self, intervention: Dict) -> str:
        """Generate synergistic opportunities"""
        synergies = [
            "Can be combined with road resurfacing",
            "Opportunity for additional safety signage",
            "Potential for landscaping improvements",
            "Integration with smart city initiatives"
        ]
        return "\n".join([f"- {synergy}" for synergy in synergies])
    
    def _generate_risk_assessment(self, intervention: Dict) -> str:
        """Generate risk assessment"""
        risks = [
            "Low risk of implementation delays",
            "Minimal environmental impact",
            "Low maintenance requirements",
            "High durability expected"
        ]
        return "\n".join([f"- {risk}" for risk in risks])
    
    def _generate_mitigation_strategies(self, intervention: Dict) -> str:
        """Generate mitigation strategies"""
        strategies = [
            "Staged implementation to minimize disruption",
            "Community engagement and communication",
            "Alternative route planning",
            "Regular progress monitoring"
        ]
        return "\n".join([f"- {strategy}" for strategy in strategies])
    
    def _generate_sensitivity_analysis(self) -> str:
        """Generate sensitivity analysis"""
        return "Analysis shows intervention remains cost-effective even with 20% cost overrun or 30% reduction in benefits."
    
    def _generate_recommendation(self, bcr: float) -> str:
        """Generate recommendation based on BCR"""
        if bcr > 2.0:
            return "Strongly recommended - High return on investment"
        elif bcr > 1.5:
            return "Recommended - Good return on investment"
        elif bcr > 1.0:
            return "Consider implementation - Positive return"
        else:
            return "Not recommended - Poor return on investment"
    
    def _generate_phase1_tasks(self) -> str:
        """Generate Phase 1 tasks"""
        tasks = [
            "Site survey and measurements",
            "Traffic study and impact assessment",
            "Permit applications and approvals",
            "Material procurement",
            "Contractor selection and mobilization"
        ]
        return "\n".join([f"- {task}" for task in tasks])
    
    def _generate_phase2_tasks(self) -> str:
        """Generate Phase 2 tasks"""
        tasks = [
            "Site preparation and marking",
            "Installation of primary intervention",
            "Quality control inspections",
            "Safety compliance verification",
            "Progress documentation"
        ]
        return "\n".join([f"- {task}" for task in tasks])
    
    def _generate_phase3_tasks(self) -> str:
        """Generate Phase 3 tasks"""
        tasks = [
            "Final inspection and testing",
            "Performance validation",
            "Documentation completion",
            "Handover to maintenance team",
            "Post-implementation monitoring"
        ]
        return "\n".join([f"- {task}" for task in tasks])
    
    def _generate_materials_list(self, intervention: Dict) -> str:
        """Generate materials list"""
        materials = [
            "Primary materials as per specifications",
            "Safety equipment and barriers",
            "Marking materials",
            "Hardware and fasteners",
            "Quality control materials"
        ]
        return "\n".join([f"- {material}" for material in materials])
    
    def _generate_labor_requirements(self) -> str:
        """Generate labor requirements"""
        return "Skilled technicians: 2-3, General laborers: 3-5, Safety supervisor: 1"
    
    def _generate_equipment_list(self) -> str:
        """Generate equipment list"""
        equipment = [
            "Excavation equipment",
            "Installation tools",
            "Safety equipment",
            "Measurement instruments",
            "Transport vehicles"
        ]
        return "\n".join([f"- {eq}" for eq in equipment])
    
    def _generate_qc_checkpoints(self) -> str:
        """Generate QC checkpoints"""
        checkpoints = [
            "Material quality verification",
            "Installation accuracy check",
            "Safety compliance inspection",
            "Performance testing",
            "Final acceptance criteria"
        ]
        return "\n".join([f"- {checkpoint}" for checkpoint in checkpoints])
    
    def _generate_compliance_checks(self) -> str:
        """Generate compliance checks"""
        checks = [
            "IRC standards compliance",
            "MoRTH guidelines adherence",
            "Local authority requirements",
            "Environmental regulations",
            "Safety standards verification"
        ]
        return "\n".join([f"- {check}" for check in checks])
    
    def _generate_risk_mitigation(self) -> str:
        """Generate risk mitigation"""
        risks = [
            "Weather contingency planning",
            "Traffic management protocols",
            "Emergency response procedures",
            "Quality assurance measures",
            "Stakeholder communication plan"
        ]
        return "\n".join([f"- {risk}" for risk in risks])
    
    def _generate_success_metrics(self) -> str:
        """Generate success metrics"""
        metrics = [
            "Accident reduction percentage",
            "Traffic flow improvement",
            "User satisfaction scores",
            "Compliance with specifications",
            "Timeline adherence"
        ]
        return "\n".join([f"- {metric}" for metric in metrics])
    
    def _generate_contributing_factors(self, accident: Dict) -> str:
        """Generate contributing factors"""
        factors = accident.get('contributing_factors', [])
        if not factors:
            factors = ["Speeding", "Poor visibility", "Driver distraction", "Road condition"]
        return "\n".join([f"- {factor}" for factor in factors])
    
    def _generate_root_causes(self, accident: Dict) -> str:
        """Generate root causes"""
        causes = [
            "Inadequate safety infrastructure",
            "Poor road design",
            "Insufficient enforcement",
            "Lack of driver education",
            "Maintenance deficiencies"
        ]
        return "\n".join([f"- {cause}" for cause in causes])
    
    def _generate_prevention_strategies(self, accident: Dict) -> str:
        """Generate prevention strategies"""
        strategies = [
            "Install appropriate safety interventions",
            "Improve road design and maintenance",
            "Enhance traffic enforcement",
            "Implement driver education programs",
            "Regular safety audits"
        ]
        return "\n".join([f"- {strategy}" for strategy in strategies])
    
    def _generate_intervention_recommendations(self, accident: Dict) -> str:
        """Generate intervention recommendations"""
        missing = accident['interventions']['missing']
        if missing:
            return "\n".join([f"- Install {intervention}" for intervention in missing[:3]])
        else:
            return "- Conduct detailed safety assessment\n- Implement additional safety measures\n- Enhance existing interventions"
    
    def _generate_policy_implications(self, accident: Dict) -> str:
        """Generate policy implications"""
        implications = [
            "Review and update safety standards",
            "Strengthen enforcement mechanisms",
            "Improve data collection systems",
            "Enhance inter-agency coordination",
            "Increase public awareness campaigns"
        ]
        return "\n".join([f"- {implication}" for implication in implications])
    
    def _generate_lessons_learned(self, accident: Dict) -> str:
        """Generate lessons learned"""
        lessons = [
            "Importance of proactive safety measures",
            "Need for regular infrastructure maintenance",
            "Value of comprehensive data analysis",
            "Critical role of stakeholder coordination",
            "Significance of continuous monitoring"
        ]
        return "\n".join([f"- {lesson}" for lesson in lessons])
    
    def generate_all_training_data(self) -> List[Dict]:
        """Generate comprehensive training dataset"""
        logger.info("Generating comprehensive training dataset...")
        
        all_examples = []
        
        # Generate different types of training examples
        logger.info("Generating intervention recommendations...")
        all_examples.extend(self.generate_intervention_recommendations(2000))
        
        logger.info("Generating cascading effects...")
        all_examples.extend(self.generate_cascading_effects(1500))
        
        logger.info("Generating cost-benefit analyses...")
        all_examples.extend(self.generate_cost_benefit_analysis(1500))
        
        logger.info("Generating implementation plans...")
        all_examples.extend(self.generate_implementation_plans(1000))
        
        logger.info("Generating accident analyses...")
        all_examples.extend(self.generate_accident_analyses(1000))
        
        # Shuffle examples
        random.shuffle(all_examples)
        
        logger.info(f"Generated {len(all_examples)} training examples")
        return all_examples
    
    def save_training_data(self, examples: List[Dict], filename: str = "llm_training_data.jsonl"):
        """Save training data in JSONL format"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Training data saved to: {output_path}")
        
        # Also save as JSON for easy inspection
        json_path = self.output_dir / filename.replace('.jsonl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training data also saved as JSON to: {json_path}")
        
        # Generate statistics
        self._generate_training_stats(examples)
    
    def _generate_training_stats(self, examples: List[Dict]):
        """Generate training data statistics"""
        stats = {
            "total_examples": len(examples),
            "instruction_types": {},
            "avg_input_length": 0,
            "avg_output_length": 0,
            "generated_at": datetime.now().isoformat()
        }
        
        total_input_length = 0
        total_output_length = 0
        
        for example in examples:
            instruction = example['instruction']
            stats['instruction_types'][instruction] = stats['instruction_types'].get(instruction, 0) + 1
            
            total_input_length += len(example['input'])
            total_output_length += len(example['output'])
        
        stats['avg_input_length'] = total_input_length / len(examples)
        stats['avg_output_length'] = total_output_length / len(examples)
        
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training statistics saved to: {stats_path}")
        logger.info(f"Total examples: {stats['total_examples']}")
        logger.info(f"Average input length: {stats['avg_input_length']:.1f} characters")
        logger.info(f"Average output length: {stats['avg_output_length']:.1f} characters")

async def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    
    generator = RoadSafetyTrainingDataGenerator()
    
    # Generate training data
    examples = generator.generate_all_training_data()
    
    # Save training data
    generator.save_training_data(examples)
    
    print(f"\nTraining data generation completed!")
    print(f"Generated {len(examples)} examples")
    print(f"Saved to: data/training/llm_training_data.jsonl")

if __name__ == "__main__":
    asyncio.run(main())
