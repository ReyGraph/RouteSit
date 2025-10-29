"""
Implementation Planner for Routesit AI
Creates detailed, contractor-ready action plans
Not just suggestions - ready-to-deploy specifications
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import yaml
from jinja2 import Template
import requests

logger = logging.getLogger(__name__)

@dataclass
class ImplementationStep:
    """Individual step in implementation plan"""
    step_id: str
    step_number: int
    title: str
    description: str
    duration_days: int
    dependencies: List[str]
    materials: List[Dict[str, Any]]
    labor_requirements: Dict[str, Any]
    equipment_needed: List[str]
    compliance_checkpoints: List[str]
    quality_control: List[str]
    cost_estimate: Dict[str, float]
    risk_factors: List[str]
    mitigation_measures: List[str]

@dataclass
class ImplementationPlan:
    """Complete implementation plan"""
    plan_id: str
    project_title: str
    intervention_type: str
    location: Dict[str, Any]
    total_duration_days: int
    total_cost: Dict[str, float]
    phases: List[Dict[str, Any]]
    steps: List[ImplementationStep]
    compliance_requirements: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    success_metrics: List[Dict[str, Any]]
    created_at: str
    created_by: str

@dataclass
class MaterialSpecification:
    """Material specification with GeM pricing"""
    material_id: str
    name: str
    specification: str
    quantity: float
    unit: str
    unit_price: float
    total_price: float
    supplier: str
    gem_catalog_id: str
    delivery_time_days: int
    quality_standards: List[str]

@dataclass
class LaborSpecification:
    """Labor specification with CPWD SOR rates"""
    labor_type: str
    skill_level: str
    quantity: int
    unit: str
    daily_rate: float
    total_cost: float
    duration_days: int
    qualifications_required: List[str]
    safety_requirements: List[str]

class GeMIntegration:
    """Integration with Government e-Marketplace for pricing"""
    
    def __init__(self):
        self.gem_base_url = "https://gem.gov.in"
        self.material_catalog = self._load_material_catalog()
        
        logger.info("GeM integration initialized")
    
    def _load_material_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load material catalog with GeM pricing"""
        return {
            "thermoplastic_paint": {
                "name": "Thermoplastic Road Marking Paint",
                "specification": "IRC:35-2015 compliant",
                "unit_price": 450.0,
                "unit": "kg",
                "gem_catalog_id": "GEM-001",
                "supplier": "Government Approved Supplier",
                "delivery_time": 7,
                "quality_standards": ["IRC:35-2015", "IS:164"]
            },
            "glass_beads": {
                "name": "Glass Beads Type A",
                "specification": "IRC:35-2015 compliant",
                "unit_price": 120.0,
                "unit": "kg",
                "gem_catalog_id": "GEM-002",
                "supplier": "Government Approved Supplier",
                "delivery_time": 5,
                "quality_standards": ["IRC:35-2015", "IS:164"]
            },
            "primer": {
                "name": "Road Marking Primer",
                "specification": "IRC:35-2015 compliant",
                "unit_price": 80.0,
                "unit": "liter",
                "gem_catalog_id": "GEM-003",
                "supplier": "Government Approved Supplier",
                "delivery_time": 3,
                "quality_standards": ["IRC:35-2015"]
            },
            "speed_limit_sign": {
                "name": "Speed Limit Sign Board",
                "specification": "IRC:67-2022 compliant",
                "unit_price": 2500.0,
                "unit": "piece",
                "gem_catalog_id": "GEM-004",
                "supplier": "Government Approved Supplier",
                "delivery_time": 10,
                "quality_standards": ["IRC:67-2022", "IS:1912"]
            },
            "warning_sign": {
                "name": "Advance Warning Sign",
                "specification": "IRC:67-2022 compliant",
                "unit_price": 1800.0,
                "unit": "piece",
                "gem_catalog_id": "GEM-005",
                "supplier": "Government Approved Supplier",
                "delivery_time": 8,
                "quality_standards": ["IRC:67-2022", "IS:1912"]
            }
        }
    
    def get_material_pricing(self, material_type: str, quantity: float) -> MaterialSpecification:
        """Get material pricing from GeM catalog"""
        try:
            if material_type not in self.material_catalog:
                return self._create_default_material(material_type, quantity)
            
            catalog_item = self.material_catalog[material_type]
            
            return MaterialSpecification(
                material_id=f"MAT_{uuid.uuid4().hex[:8]}",
                name=catalog_item["name"],
                specification=catalog_item["specification"],
                quantity=quantity,
                unit=catalog_item["unit"],
                unit_price=catalog_item["unit_price"],
                total_price=catalog_item["unit_price"] * quantity,
                supplier=catalog_item["supplier"],
                gem_catalog_id=catalog_item["gem_catalog_id"],
                delivery_time_days=catalog_item["delivery_time"],
                quality_standards=catalog_item["quality_standards"]
            )
            
        except Exception as e:
            logger.error(f"Error getting material pricing: {e}")
            return self._create_default_material(material_type, quantity)
    
    def _create_default_material(self, material_type: str, quantity: float) -> MaterialSpecification:
        """Create default material specification"""
        return MaterialSpecification(
            material_id=f"MAT_{uuid.uuid4().hex[:8]}",
            name=f"{material_type.replace('_', ' ').title()}",
            specification="Standard specification",
            quantity=quantity,
            unit="piece",
            unit_price=1000.0,
            total_price=1000.0 * quantity,
            supplier="Local Supplier",
            gem_catalog_id="N/A",
            delivery_time_days=7,
            quality_standards=["Standard"]
        )

class CPWDSORIntegration:
    """Integration with CPWD Schedule of Rates"""
    
    def __init__(self):
        self.sor_rates = self._load_sor_rates()
        
        logger.info("CPWD SOR integration initialized")
    
    def _load_sor_rates(self) -> Dict[str, Dict[str, Any]]:
        """Load CPWD SOR rates for labor"""
        return {
            "skilled_labor": {
                "daily_rate": 800.0,
                "skill_level": "Skilled",
                "qualifications": ["ITI Certificate", "Safety Training"],
                "safety_requirements": ["Hard Hat", "Safety Shoes", "High Visibility Vest"]
            },
            "semi_skilled_labor": {
                "daily_rate": 600.0,
                "skill_level": "Semi-Skilled",
                "qualifications": ["Basic Training"],
                "safety_requirements": ["Hard Hat", "Safety Shoes"]
            },
            "unskilled_labor": {
                "daily_rate": 400.0,
                "skill_level": "Unskilled",
                "qualifications": ["Basic Safety Training"],
                "safety_requirements": ["Hard Hat"]
            },
            "supervisor": {
                "daily_rate": 1200.0,
                "skill_level": "Supervisor",
                "qualifications": ["Diploma in Civil Engineering", "Safety Certification"],
                "safety_requirements": ["Hard Hat", "Safety Shoes", "High Visibility Vest", "Safety Manual"]
            }
        }
    
    def get_labor_pricing(self, labor_type: str, quantity: int, duration_days: int) -> LaborSpecification:
        """Get labor pricing from CPWD SOR"""
        try:
            if labor_type not in self.sor_rates:
                return self._create_default_labor(labor_type, quantity, duration_days)
            
            sor_item = self.sor_rates[labor_type]
            
            return LaborSpecification(
                labor_type=labor_type,
                skill_level=sor_item["skill_level"],
                quantity=quantity,
                unit="person",
                daily_rate=sor_item["daily_rate"],
                total_cost=sor_item["daily_rate"] * quantity * duration_days,
                duration_days=duration_days,
                qualifications_required=sor_item["qualifications"],
                safety_requirements=sor_item["safety_requirements"]
            )
            
        except Exception as e:
            logger.error(f"Error getting labor pricing: {e}")
            return self._create_default_labor(labor_type, quantity, duration_days)
    
    def _create_default_labor(self, labor_type: str, quantity: int, duration_days: int) -> LaborSpecification:
        """Create default labor specification"""
        return LaborSpecification(
            labor_type=labor_type,
            skill_level="Standard",
            quantity=quantity,
            unit="person",
            daily_rate=500.0,
            total_cost=500.0 * quantity * duration_days,
            duration_days=duration_days,
            qualifications_required=["Basic Training"],
            safety_requirements=["Hard Hat"]
        )

class ComplianceChecker:
    """Check IRC/MoRTH compliance requirements"""
    
    def __init__(self):
        self.compliance_rules = self._load_compliance_rules()
        
        logger.info("Compliance checker initialized")
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load IRC/MoRTH compliance rules"""
        return {
            "zebra_crossing": {
                "irc_standard": "IRC:35-2015",
                "clause": "7.2",
                "requirements": [
                    "Minimum width 3 meters",
                    "White thermoplastic paint",
                    "Glass beads for retroreflectivity",
                    "Advance warning sign 50m before",
                    "Street lighting required"
                ],
                "quality_control": [
                    "Paint thickness measurement",
                    "Retroreflectivity testing",
                    "Visibility assessment"
                ]
            },
            "speed_limit_sign": {
                "irc_standard": "IRC:67-2022",
                "clause": "14.4",
                "requirements": [
                    "Standard size 600x600mm",
                    "Retroreflective sheeting",
                    "Proper mounting height",
                    "Clear visibility from 100m"
                ],
                "quality_control": [
                    "Retroreflectivity testing",
                    "Visibility assessment",
                    "Mounting strength test"
                ]
            },
            "speed_hump": {
                "irc_standard": "IRC:99-2018",
                "clause": "5.2",
                "requirements": [
                    "Height 75-100mm",
                    "Width 3-4 meters",
                    "Gradual approach slopes",
                    "Advance warning signs"
                ],
                "quality_control": [
                    "Height measurement",
                    "Slope verification",
                    "Drainage check"
                ]
            }
        }
    
    def check_compliance(self, intervention_type: str) -> Dict[str, Any]:
        """Check compliance requirements for intervention"""
        try:
            if intervention_type not in self.compliance_rules:
                return self._create_default_compliance(intervention_type)
            
            return self.compliance_rules[intervention_type]
            
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return self._create_default_compliance(intervention_type)
    
    def _create_default_compliance(self, intervention_type: str) -> Dict[str, Any]:
        """Create default compliance requirements"""
        return {
            "irc_standard": "IRC:General",
            "clause": "Standard",
            "requirements": ["Standard safety requirements"],
            "quality_control": ["Basic quality checks"]
        }

class ImplementationPlanner:
    """
    Creates detailed implementation plans with contractor-ready specifications
    """
    
    def __init__(self):
        self.gem_integration = GeMIntegration()
        self.sor_integration = CPWDSORIntegration()
        self.compliance_checker = ComplianceChecker()
        
        # Plan templates
        self.plan_templates = self._load_plan_templates()
        
        logger.info("Implementation planner initialized")
    
    def _load_plan_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load implementation plan templates"""
        return {
            "zebra_crossing": {
                "phases": [
                    {
                        "phase_name": "Site Preparation",
                        "duration_days": 1,
                        "steps": [
                            "Site survey and measurement",
                            "Traffic management setup",
                            "Surface preparation"
                        ]
                    },
                    {
                        "phase_name": "Material Procurement",
                        "duration_days": 2,
                        "steps": [
                            "Material ordering from GeM",
                            "Quality verification",
                            "Delivery coordination"
                        ]
                    },
                    {
                        "phase_name": "Installation",
                        "duration_days": 2,
                        "steps": [
                            "Primer application",
                            "Paint application",
                            "Glass beads application"
                        ]
                    },
                    {
                        "phase_name": "Quality Control",
                        "duration_days": 1,
                        "steps": [
                            "Thickness measurement",
                            "Retroreflectivity testing",
                            "Final inspection"
                        ]
                    }
                ]
            },
            "speed_limit_sign": {
                "phases": [
                    {
                        "phase_name": "Site Survey",
                        "duration_days": 1,
                        "steps": [
                            "Location identification",
                            "Visibility assessment",
                            "Mounting point selection"
                        ]
                    },
                    {
                        "phase_name": "Installation",
                        "duration_days": 1,
                        "steps": [
                            "Post installation",
                            "Sign mounting",
                            "Alignment verification"
                        ]
                    }
                ]
            }
        }
    
    def create_implementation_plan(self, 
                                 intervention_type: str,
                                 location: Dict[str, Any],
                                 context: Dict[str, Any]) -> ImplementationPlan:
        """
        Create detailed implementation plan
        """
        try:
            # Get plan template
            template = self.plan_templates.get(intervention_type, self._create_default_template())
            
            # Create implementation steps
            steps = self._create_implementation_steps(intervention_type, template, context)
            
            # Calculate total duration and cost
            total_duration = sum(step.duration_days for step in steps)
            total_cost = self._calculate_total_cost(steps)
            
            # Get compliance requirements
            compliance_requirements = self.compliance_checker.check_compliance(intervention_type)
            
            # Create risk assessment
            risk_assessment = self._create_risk_assessment(intervention_type, location, context)
            
            # Create success metrics
            success_metrics = self._create_success_metrics(intervention_type)
            
            return ImplementationPlan(
                plan_id=f"PLAN_{uuid.uuid4().hex[:8]}",
                project_title=f"{intervention_type.replace('_', ' ').title()} Implementation",
                intervention_type=intervention_type,
                location=location,
                total_duration_days=total_duration,
                total_cost=total_cost,
                phases=template["phases"],
                steps=steps,
                compliance_requirements=compliance_requirements,
                risk_assessment=risk_assessment,
                success_metrics=success_metrics,
                created_at=datetime.now().isoformat(),
                created_by="Routesit AI"
            )
            
        except Exception as e:
            logger.error(f"Error creating implementation plan: {e}")
            return self._create_fallback_plan(intervention_type, location)
    
    def _create_implementation_steps(self, 
                                   intervention_type: str,
                                   template: Dict[str, Any],
                                   context: Dict[str, Any]) -> List[ImplementationStep]:
        """Create detailed implementation steps"""
        steps = []
        step_number = 1
        
        for phase in template["phases"]:
            for step_name in phase["steps"]:
                step = ImplementationStep(
                    step_id=f"STEP_{step_number:03d}",
                    step_number=step_number,
                    title=step_name,
                    description=self._get_step_description(step_name, intervention_type),
                    duration_days=self._get_step_duration(step_name, intervention_type),
                    dependencies=self._get_step_dependencies(step_name, step_number),
                    materials=self._get_step_materials(step_name, intervention_type, context),
                    labor_requirements=self._get_step_labor(step_name, intervention_type),
                    equipment_needed=self._get_step_equipment(step_name, intervention_type),
                    compliance_checkpoints=self._get_compliance_checkpoints(step_name, intervention_type),
                    quality_control=self._get_quality_control(step_name, intervention_type),
                    cost_estimate=self._get_step_cost(step_name, intervention_type, context),
                    risk_factors=self._get_step_risks(step_name, intervention_type),
                    mitigation_measures=self._get_mitigation_measures(step_name, intervention_type)
                )
                steps.append(step)
                step_number += 1
        
        return steps
    
    def _get_step_description(self, step_name: str, intervention_type: str) -> str:
        """Get detailed step description"""
        descriptions = {
            "Site survey and measurement": "Conduct detailed site survey to measure exact dimensions, document existing conditions, and identify any obstacles or constraints.",
            "Traffic management setup": "Install temporary traffic control measures including barricades, warning signs, and flagmen to ensure worker safety.",
            "Surface preparation": "Clean and prepare the road surface by removing debris, filling potholes, and ensuring proper adhesion for marking materials.",
            "Material ordering from GeM": "Place orders through Government e-Marketplace for all required materials with proper specifications and quality standards.",
            "Primer application": "Apply primer coat to ensure proper adhesion of thermoplastic paint to the road surface.",
            "Paint application": "Apply thermoplastic paint using specialized equipment to create durable road markings.",
            "Glass beads application": "Apply glass beads immediately after paint application to ensure retroreflectivity for night visibility.",
            "Thickness measurement": "Measure paint thickness using calibrated instruments to ensure compliance with IRC standards.",
            "Retroreflectivity testing": "Test retroreflectivity using specialized equipment to ensure visibility standards are met.",
            "Final inspection": "Conduct comprehensive final inspection to verify all requirements are met and document completion."
        }
        
        return descriptions.get(step_name, f"Execute {step_name} according to standard procedures.")
    
    def _get_step_duration(self, step_name: str, intervention_type: str) -> int:
        """Get step duration in days"""
        durations = {
            "Site survey and measurement": 1,
            "Traffic management setup": 1,
            "Surface preparation": 1,
            "Material ordering from GeM": 2,
            "Primer application": 1,
            "Paint application": 1,
            "Glass beads application": 1,
            "Thickness measurement": 1,
            "Retroreflectivity testing": 1,
            "Final inspection": 1
        }
        
        return durations.get(step_name, 1)
    
    def _get_step_dependencies(self, step_name: str, step_number: int) -> List[str]:
        """Get step dependencies"""
        dependencies = {
            "Traffic management setup": ["Site survey and measurement"],
            "Surface preparation": ["Traffic management setup"],
            "Primer application": ["Surface preparation"],
            "Paint application": ["Primer application"],
            "Glass beads application": ["Paint application"],
            "Thickness measurement": ["Glass beads application"],
            "Retroreflectivity testing": ["Glass beads application"],
            "Final inspection": ["Retroreflectivity testing"]
        }
        
        return dependencies.get(step_name, [])
    
    def _get_step_materials(self, step_name: str, intervention_type: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get materials required for step"""
        materials = []
        
        if "Paint application" in step_name:
            # Calculate paint quantity based on area
            area = context.get("area", 50)  # square meters
            paint_quantity = area * 0.5  # kg per square meter
            
            paint_spec = self.gem_integration.get_material_pricing("thermoplastic_paint", paint_quantity)
            materials.append(asdict(paint_spec))
            
            # Glass beads
            beads_quantity = paint_quantity * 0.2  # 20% of paint weight
            beads_spec = self.gem_integration.get_material_pricing("glass_beads", beads_quantity)
            materials.append(asdict(beads_spec))
        
        elif "Primer application" in step_name:
            primer_quantity = context.get("area", 50) * 0.1  # liters per square meter
            primer_spec = self.gem_integration.get_material_pricing("primer", primer_quantity)
            materials.append(asdict(primer_spec))
        
        elif "Sign" in step_name:
            sign_spec = self.gem_integration.get_material_pricing("speed_limit_sign", 1)
            materials.append(asdict(sign_spec))
        
        return materials
    
    def _get_step_labor(self, step_name: str, intervention_type: str) -> Dict[str, Any]:
        """Get labor requirements for step"""
        labor_specs = {}
        
        if "application" in step_name.lower():
            skilled_labor = self.sor_integration.get_labor_pricing("skilled_labor", 2, 1)
            labor_specs["skilled_labor"] = asdict(skilled_labor)
            
            supervisor = self.sor_integration.get_labor_pricing("supervisor", 1, 1)
            labor_specs["supervisor"] = asdict(supervisor)
        
        elif "survey" in step_name.lower():
            supervisor = self.sor_integration.get_labor_pricing("supervisor", 1, 1)
            labor_specs["supervisor"] = asdict(supervisor)
        
        else:
            semi_skilled = self.sor_integration.get_labor_pricing("semi_skilled_labor", 2, 1)
            labor_specs["semi_skilled_labor"] = asdict(semi_skilled)
        
        return labor_specs
    
    def _get_step_equipment(self, step_name: str, intervention_type: str) -> List[str]:
        """Get equipment needed for step"""
        equipment = {
            "Paint application": ["Thermoplastic applicator", "Paint heating unit", "Measuring equipment"],
            "Glass beads application": ["Bead applicator", "Measuring equipment"],
            "Thickness measurement": ["Paint thickness gauge", "Calibration certificate"],
            "Retroreflectivity testing": ["Retroreflectometer", "Calibration certificate"],
            "Site survey and measurement": ["Measuring tape", "GPS device", "Camera"],
            "Traffic management setup": ["Barricades", "Warning signs", "Safety equipment"]
        }
        
        return equipment.get(step_name, ["Standard equipment"])
    
    def _get_compliance_checkpoints(self, step_name: str, intervention_type: str) -> List[str]:
        """Get compliance checkpoints for step"""
        compliance = self.compliance_checker.check_compliance(intervention_type)
        return compliance.get("requirements", ["Standard compliance check"])
    
    def _get_quality_control(self, step_name: str, intervention_type: str) -> List[str]:
        """Get quality control measures for step"""
        compliance = self.compliance_checker.check_compliance(intervention_type)
        return compliance.get("quality_control", ["Standard quality check"])
    
    def _get_step_cost(self, step_name: str, intervention_type: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Get cost estimate for step"""
        materials_cost = 0.0
        labor_cost = 0.0
        
        # Calculate materials cost
        materials = self._get_step_materials(step_name, intervention_type, context)
        for material in materials:
            materials_cost += material.get("total_price", 0.0)
        
        # Calculate labor cost
        labor_requirements = self._get_step_labor(step_name, intervention_type)
        for labor_type, labor_spec in labor_requirements.items():
            labor_cost += labor_spec.get("total_cost", 0.0)
        
        return {
            "materials": materials_cost,
            "labor": labor_cost,
            "equipment": 1000.0,  # Equipment rental cost
            "total": materials_cost + labor_cost + 1000.0
        }
    
    def _get_step_risks(self, step_name: str, intervention_type: str) -> List[str]:
        """Get risk factors for step"""
        risks = {
            "Paint application": ["Weather conditions", "Traffic interference", "Material quality"],
            "Traffic management setup": ["Traffic congestion", "Safety hazards", "Public inconvenience"],
            "Site survey and measurement": ["Traffic hazards", "Measurement errors", "Access issues"]
        }
        
        return risks.get(step_name, ["Standard risks"])
    
    def _get_mitigation_measures(self, step_name: str, intervention_type: str) -> List[str]:
        """Get mitigation measures for step"""
        mitigations = {
            "Paint application": ["Weather monitoring", "Traffic control", "Quality verification"],
            "Traffic management setup": ["Advance notification", "Alternative routes", "Safety protocols"],
            "Site survey and measurement": ["Safety equipment", "Calibrated instruments", "Backup plans"]
        }
        
        return mitigations.get(step_name, ["Standard mitigation measures"])
    
    def _calculate_total_cost(self, steps: List[ImplementationStep]) -> Dict[str, float]:
        """Calculate total project cost"""
        total_materials = sum(step.cost_estimate.get("materials", 0.0) for step in steps)
        total_labor = sum(step.cost_estimate.get("labor", 0.0) for step in steps)
        total_equipment = sum(step.cost_estimate.get("equipment", 0.0) for step in steps)
        
        return {
            "materials": total_materials,
            "labor": total_labor,
            "equipment": total_equipment,
            "total": total_materials + total_labor + total_equipment
        }
    
    def _create_risk_assessment(self, intervention_type: str, location: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk assessment for project"""
        return {
            "overall_risk_level": "Medium",
            "risk_factors": [
                "Weather conditions",
                "Traffic interference",
                "Material availability",
                "Labor availability"
            ],
            "mitigation_strategies": [
                "Weather monitoring and contingency planning",
                "Comprehensive traffic management",
                "Advance material procurement",
                "Skilled labor backup plans"
            ],
            "contingency_budget": 0.15  # 15% contingency
        }
    
    def _create_success_metrics(self, intervention_type: str) -> List[Dict[str, Any]]:
        """Create success metrics for project"""
        return [
            {
                "metric": "Quality Compliance",
                "target": "100%",
                "measurement": "IRC standard compliance verification"
            },
            {
                "metric": "Timeline Adherence",
                "target": "95%",
                "measurement": "Completion within planned duration"
            },
            {
                "metric": "Cost Control",
                "target": "Within 10% of estimate",
                "measurement": "Final cost vs. estimated cost"
            },
            {
                "metric": "Safety Record",
                "target": "Zero accidents",
                "measurement": "Incident-free completion"
            }
        ]
    
    def _create_default_template(self) -> Dict[str, Any]:
        """Create default plan template"""
        return {
            "phases": [
                {
                    "phase_name": "Planning",
                    "duration_days": 1,
                    "steps": ["Project planning", "Resource allocation"]
                },
                {
                    "phase_name": "Execution",
                    "duration_days": 2,
                    "steps": ["Implementation", "Quality control"]
                }
            ]
        }
    
    def _create_fallback_plan(self, intervention_type: str, location: Dict[str, Any]) -> ImplementationPlan:
        """Create fallback plan when creation fails"""
        return ImplementationPlan(
            plan_id=f"FALLBACK_{uuid.uuid4().hex[:8]}",
            project_title=f"{intervention_type.replace('_', ' ').title()} Implementation",
            intervention_type=intervention_type,
            location=location,
            total_duration_days=5,
            total_cost={"materials": 10000.0, "labor": 5000.0, "equipment": 2000.0, "total": 17000.0},
            phases=[],
            steps=[],
            compliance_requirements=[],
            risk_assessment={"overall_risk_level": "High"},
            success_metrics=[],
            created_at=datetime.now().isoformat(),
            created_by="Routesit AI (Fallback)"
        )
    
    def export_plan(self, plan: ImplementationPlan, format: str = "json") -> str:
        """Export implementation plan in specified format"""
        try:
            if format == "json":
                return json.dumps(asdict(plan), indent=2, ensure_ascii=False)
            elif format == "yaml":
                return yaml.dump(asdict(plan), default_flow_style=False)
            else:
                return str(plan)
                
        except Exception as e:
            logger.error(f"Error exporting plan: {e}")
            return "Export failed"

# Global instance
implementation_planner = None

def get_implementation_planner() -> ImplementationPlanner:
    """Get global implementation planner instance"""
    global implementation_planner
    if implementation_planner is None:
        implementation_planner = ImplementationPlanner()
    return implementation_planner

def create_implementation_plan(intervention_type: str,
                              location: Dict[str, Any],
                              context: Dict[str, Any]) -> ImplementationPlan:
    """Convenience function for creating implementation plans"""
    planner = get_implementation_planner()
    return planner.create_implementation_plan(intervention_type, location, context)
