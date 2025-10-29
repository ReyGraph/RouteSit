#!/usr/bin/env python3
"""
YOLOv8 Integration with DriveIndia/IDD-D Pre-trained Weights
Road infrastructure detection for multi-modal analysis
"""

import os
import sys
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import torch

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Result from YOLO detection"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int

@dataclass
class RoadInfrastructureAnalysis:
    """Complete road infrastructure analysis"""
    detected_objects: List[DetectionResult]
    road_signs: List[DetectionResult]
    road_markings: List[DetectionResult]
    vehicles: List[DetectionResult]
    pedestrians: List[DetectionResult]
    infrastructure_quality: Dict[str, Any]
    safety_assessment: Dict[str, Any]
    recommendations: List[str]

class YOLOv8Detector:
    """YOLOv8 detector for Indian road infrastructure"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._find_model()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Indian road infrastructure classes
        self.infrastructure_classes = {
            # DriveIndia classes (24 categories)
            "car": "vehicle",
            "truck": "vehicle", 
            "bus": "vehicle",
            "motorcycle": "vehicle",
            "bicycle": "vehicle",
            "autorickshaw": "vehicle",
            "person": "pedestrian",
            "rider": "pedestrian",
            "traffic_sign": "road_sign",
            "traffic_light": "traffic_control",
            "pole": "infrastructure",
            "tree": "infrastructure",
            "building": "infrastructure",
            "fence": "infrastructure",
            "bridge": "infrastructure",
            "tunnel": "infrastructure",
            "sidewalk": "infrastructure",
            "road": "road_surface",
            "lane_marking": "road_marking",
            "zebra_crossing": "road_marking",
            "stop_line": "road_marking",
            "speed_bump": "traffic_calming",
            "guard_rail": "safety_barrier",
            "animal": "hazard"
        }
        
        # IDD-D classes (15 categories)
        self.idd_classes = {
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle", 
            "motorcycle": "vehicle",
            "bicycle": "vehicle",
            "autorickshaw": "vehicle",
            "person": "pedestrian",
            "rider": "pedestrian",
            "traffic_sign": "road_sign",
            "traffic_light": "traffic_control",
            "pole": "infrastructure",
            "tree": "infrastructure",
            "building": "infrastructure",
            "fence": "infrastructure",
            "sidewalk": "infrastructure"
        }
        
        self._load_model()
    
    def _find_model(self) -> str:
        """Find available YOLOv8 model"""
        model_dir = Path("models/yolov8")
        
        # Look for pre-trained models
        model_patterns = [
            "yolov8n.pt",  # Nano
            "yolov8s.pt",  # Small
            "yolov8m.pt",  # Medium
            "yolov8l.pt",  # Large
            "yolov8x.pt",  # Extra Large
            "best.pt",     # Custom trained
            "*.pt"
        ]
        
        for pattern in model_patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                return str(matches[0])
        
        # Default to YOLOv8n if no model found
        return "yolov8n.pt"
    
    def _load_model(self):
        """Load YOLOv8 model"""
        if not YOLO:
            logger.error("ultralytics not available")
            return
        
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            self.model = YOLO(self.model_path)
            logger.info("YOLOv8 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            self.model = None
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        """Detect objects in image"""
        if not self.model:
            logger.error("Model not loaded")
            return []
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Run detection
            results = self.model(image, conf=confidence_threshold, device=self.device)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # Map to infrastructure category
                        category = self.infrastructure_classes.get(class_name, "unknown")
                        
                        detection = DetectionResult(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            center=((x1 + x2) // 2, (y1 + y2) // 2),
                            area=(x2 - x1) * (y2 - y1)
                        )
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def analyze_road_infrastructure(self, image_path: str) -> RoadInfrastructureAnalysis:
        """Complete road infrastructure analysis"""
        
        detections = self.detect_objects(image_path)
        
        # Categorize detections
        road_signs = []
        road_markings = []
        vehicles = []
        pedestrians = []
        
        for detection in detections:
            category = self.infrastructure_classes.get(detection.class_name, "unknown")
            
            if category == "road_sign":
                road_signs.append(detection)
            elif category == "road_marking":
                road_markings.append(detection)
            elif category == "vehicle":
                vehicles.append(detection)
            elif category == "pedestrian":
                pedestrians.append(detection)
        
        # Analyze infrastructure quality
        quality_analysis = self._analyze_infrastructure_quality(detections)
        
        # Safety assessment
        safety_assessment = self._assess_safety(detections)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detections, quality_analysis, safety_assessment)
        
        return RoadInfrastructureAnalysis(
            detected_objects=detections,
            road_signs=road_signs,
            road_markings=road_markings,
            vehicles=vehicles,
            pedestrians=pedestrians,
            infrastructure_quality=quality_analysis,
            safety_assessment=safety_assessment,
            recommendations=recommendations
        )
    
    def _analyze_infrastructure_quality(self, detections: List[DetectionResult]) -> Dict[str, Any]:
        """Analyze quality of road infrastructure"""
        
        # Count different types of infrastructure
        sign_count = len([d for d in detections if self.infrastructure_classes.get(d.class_name) == "road_sign"])
        marking_count = len([d for d in detections if self.infrastructure_classes.get(d.class_name) == "road_marking"])
        vehicle_count = len([d for d in detections if self.infrastructure_classes.get(d.class_name) == "vehicle"])
        pedestrian_count = len([d for d in detections if self.infrastructure_classes.get(d.class_name) == "pedestrian"])
        
        # Calculate quality scores
        sign_quality = min(sign_count / 5.0, 1.0)  # Normalize to 0-1
        marking_quality = min(marking_count / 3.0, 1.0)
        
        # Overall quality score
        overall_quality = (sign_quality + marking_quality) / 2.0
        
        return {
            "overall_score": overall_quality,
            "sign_count": sign_count,
            "marking_count": marking_count,
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "sign_quality_score": sign_quality,
            "marking_quality_score": marking_quality,
            "assessment": "Good" if overall_quality > 0.7 else "Fair" if overall_quality > 0.4 else "Poor"
        }
    
    def _assess_safety(self, detections: List[DetectionResult]) -> Dict[str, Any]:
        """Assess road safety based on detections"""
        
        # Count safety-related objects
        safety_signs = len([d for d in detections if "sign" in d.class_name.lower()])
        traffic_lights = len([d for d in detections if "light" in d.class_name.lower()])
        crossings = len([d for d in detections if "crossing" in d.class_name.lower()])
        barriers = len([d for d in detections if "barrier" in d.class_name.lower() or "rail" in d.class_name.lower()])
        
        # Calculate safety score
        safety_score = min((safety_signs + traffic_lights + crossings + barriers) / 10.0, 1.0)
        
        # Identify hazards
        hazards = []
        if len([d for d in detections if d.class_name == "animal"]) > 0:
            hazards.append("Animals on road")
        
        high_vehicle_density = len([d for d in detections if self.infrastructure_classes.get(d.class_name) == "vehicle"]) > 5
        if high_vehicle_density and crossings == 0:
            hazards.append("High vehicle density without pedestrian crossings")
        
        return {
            "safety_score": safety_score,
            "safety_level": "High" if safety_score > 0.7 else "Medium" if safety_score > 0.4 else "Low",
            "safety_signs": safety_signs,
            "traffic_lights": traffic_lights,
            "crossings": crossings,
            "barriers": barriers,
            "hazards": hazards,
            "risk_factors": len(hazards)
        }
    
    def _generate_recommendations(self, detections: List[DetectionResult], 
                                quality_analysis: Dict[str, Any], 
                                safety_assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_analysis["sign_count"] < 2:
            recommendations.append("Install additional road signs for better guidance")
        
        if quality_analysis["marking_count"] < 1:
            recommendations.append("Add road markings for lane separation and safety")
        
        if quality_analysis["overall_score"] < 0.5:
            recommendations.append("Comprehensive infrastructure upgrade needed")
        
        # Safety-based recommendations
        if safety_assessment["crossings"] == 0 and safety_assessment["pedestrian_count"] > 0:
            recommendations.append("Install pedestrian crossing for safety")
        
        if safety_assessment["traffic_lights"] == 0 and safety_assessment["vehicle_count"] > 3:
            recommendations.append("Consider traffic signal installation")
        
        if len(safety_assessment["hazards"]) > 0:
            recommendations.append("Address identified safety hazards")
        
        if safety_assessment["safety_score"] < 0.4:
            recommendations.append("Immediate safety improvements required")
        
        return recommendations
    
    def create_visualization(self, image_path: str, analysis: RoadInfrastructureAnalysis, 
                           output_path: str = None) -> str:
        """Create visualization of detections"""
        
        if not output_path:
            output_path = f"output/detection_{Path(image_path).stem}.jpg"
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Draw detections
        for detection in analysis.detected_objects:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on category
            category = self.infrastructure_classes.get(detection.class_name, "unknown")
            if category == "road_sign":
                color = (0, 255, 0)  # Green
            elif category == "road_marking":
                color = (255, 0, 0)  # Blue
            elif category == "vehicle":
                color = (0, 0, 255)  # Red
            elif category == "pedestrian":
                color = (255, 255, 0)  # Cyan
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save visualization
        cv2.imwrite(output_path, image)
        logger.info(f"Visualization saved to: {output_path}")
        
        return output_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "is_loaded": self.model is not None,
            "infrastructure_classes": len(self.infrastructure_classes),
            "idd_classes": len(self.idd_classes)
        }

def main():
    """Test the YOLOv8 detector"""
    logging.basicConfig(level=logging.INFO)
    
    print("Testing YOLOv8 Road Infrastructure Detector...")
    
    detector = YOLOv8Detector()
    
    # Test model info
    info = detector.get_model_info()
    print(f"Model info: {info}")
    
    if not detector.model:
        print("Model not loaded. Please install ultralytics: pip install ultralytics")
        return
    
    # Test with sample image (if available)
    sample_images = [
        "data/sample_images/road_scene.jpg",
        "data/sample_images/intersection.jpg",
        "data/sample_images/school_zone.jpg"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            print(f"\nAnalyzing: {image_path}")
            
            analysis = detector.analyze_road_infrastructure(image_path)
            
            print(f"Detected objects: {len(analysis.detected_objects)}")
            print(f"Road signs: {len(analysis.road_signs)}")
            print(f"Road markings: {len(analysis.road_markings)}")
            print(f"Vehicles: {len(analysis.vehicles)}")
            print(f"Pedestrians: {len(analysis.pedestrians)}")
            
            print(f"Infrastructure quality: {analysis.infrastructure_quality['assessment']}")
            print(f"Safety level: {analysis.safety_assessment['safety_level']}")
            
            print("Recommendations:")
            for rec in analysis.recommendations:
                print(f"  - {rec}")
            
            # Create visualization
            viz_path = detector.create_visualization(image_path, analysis)
            if viz_path:
                print(f"Visualization saved to: {viz_path}")
            
            break
    else:
        print("No sample images found for testing")
        print("Please add sample road images to data/sample_images/")

if __name__ == "__main__":
    main()
