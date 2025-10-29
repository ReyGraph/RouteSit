import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

class RoadSafetyVisionAnalyzer:
    """Computer vision pipeline for road safety analysis using YOLOv8"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
        
        # Road safety specific classes
        self.road_safety_classes = {
            'traffic_sign': 0,
            'road_marking': 1,
            'pedestrian_crossing': 2,
            'speed_limit': 3,
            'stop_sign': 4,
            'warning_sign': 5,
            'information_sign': 6,
            'regulatory_sign': 7,
            'zebra_crossing': 8,
            'lane_marking': 9,
            'arrow_marking': 10,
            'speed_hump': 11,
            'guard_rail': 12,
            'traffic_light': 13,
            'bus_stop': 14,
            'school_zone': 15
        }
        
        # Problem detection patterns
        self.problem_patterns = {
            'faded': ['low_contrast', 'worn_appearance', 'reduced_visibility'],
            'damaged': ['cracks', 'missing_parts', 'deformed_shape'],
            'missing': ['absence', 'empty_space', 'no_detection'],
            'obstructed': ['blocked_view', 'vegetation', 'other_objects'],
            'improper_placement': ['wrong_position', 'incorrect_height', 'poor_visibility']
        }
    
    def _initialize_model(self):
        """Initialize YOLOv8 model"""
        try:
            logger.info(f"Initializing YOLOv8 model on {self.device}")
            self.model = YOLO(self.model_path)
            logger.info("YOLOv8 model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 model: {e}")
            self.model = None
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze road safety image and detect problems
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Analysis results with detected objects and problems
        """
        try:
            if self.model is None:
                return self._fallback_analysis(image_path)
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run YOLO detection
            results = self.model(image, conf=0.5)
            
            # Process results
            analysis_result = self._process_detection_results(results, image)
            
            # Add problem detection
            analysis_result['problems_detected'] = self._detect_problems(image, analysis_result['detections'])
            
            # Generate recommendations
            analysis_result['recommendations'] = self._generate_vision_recommendations(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._fallback_analysis(image_path)
    
    def _process_detection_results(self, results, image: np.ndarray) -> Dict[str, Any]:
        """Process YOLO detection results"""
        
        detections = []
        confidence_scores = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Map to road safety class
                    class_name = self._map_to_road_safety_class(class_id)
                    
                    detection = {
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'area': int((x2 - x1) * (y2 - y1)),
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    }
                    
                    detections.append(detection)
                    confidence_scores.append(confidence)
        
        return {
            'detections': detections,
            'total_detections': len(detections),
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'image_dimensions': [image.shape[1], image.shape[0]],  # width, height
            'analysis_timestamp': self._get_timestamp()
        }
    
    def _map_to_road_safety_class(self, class_id: int) -> str:
        """Map YOLO class ID to road safety class"""
        # This would need to be customized based on your trained model
        class_mapping = {
            0: 'traffic_sign',
            1: 'road_marking',
            2: 'pedestrian_crossing',
            3: 'speed_limit',
            4: 'stop_sign',
            5: 'warning_sign',
            6: 'information_sign',
            7: 'regulatory_sign',
            8: 'zebra_crossing',
            9: 'lane_marking',
            10: 'arrow_marking',
            11: 'speed_hump',
            12: 'guard_rail',
            13: 'traffic_light',
            14: 'bus_stop',
            15: 'school_zone'
        }
        return class_mapping.get(class_id, 'unknown')
    
    def _detect_problems(self, image: np.ndarray, detections: List[Dict]) -> List[Dict[str, Any]]:
        """Detect problems in the image based on visual analysis"""
        
        problems = []
        
        # Analyze each detection for problems
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract region of interest
            roi = image[y1:y2, x1:x2]
            
            # Analyze for specific problems
            roi_problems = self._analyze_roi_problems(roi, detection)
            problems.extend(roi_problems)
        
        # Analyze overall image for problems
        overall_problems = self._analyze_overall_problems(image, detections)
        problems.extend(overall_problems)
        
        return problems
    
    def _analyze_roi_problems(self, roi: np.ndarray, detection: Dict) -> List[Dict[str, Any]]:
        """Analyze region of interest for specific problems"""
        
        problems = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for faded appearance
        if self._is_faded(gray):
            problems.append({
                'type': 'faded',
                'object': detection['class_name'],
                'confidence': 0.8,
                'description': f"{detection['class_name']} appears faded and needs repainting",
                'severity': 'medium',
                'location': detection['bbox']
            })
        
        # Check for damage
        if self._is_damaged(roi):
            problems.append({
                'type': 'damaged',
                'object': detection['class_name'],
                'confidence': 0.7,
                'description': f"{detection['class_name']} shows signs of damage",
                'severity': 'high',
                'location': detection['bbox']
            })
        
        # Check for obstruction
        if self._is_obstructed(roi):
            problems.append({
                'type': 'obstructed',
                'object': detection['class_name'],
                'confidence': 0.6,
                'description': f"{detection['class_name']} appears to be obstructed",
                'severity': 'high',
                'location': detection['bbox']
            })
        
        return problems
    
    def _is_faded(self, gray_roi: np.ndarray) -> bool:
        """Check if ROI appears faded"""
        # Calculate contrast and brightness
        mean_brightness = np.mean(gray_roi)
        std_contrast = np.std(gray_roi)
        
        # Faded objects typically have low contrast
        return std_contrast < 30 and mean_brightness > 150
    
    def _is_damaged(self, roi: np.ndarray) -> bool:
        """Check if ROI shows signs of damage"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Look for edges that might indicate cracks or damage
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        
        # High edge density might indicate damage
        return edge_density > 0.1
    
    def _is_obstructed(self, roi: np.ndarray) -> bool:
        """Check if ROI is obstructed"""
        # Simple obstruction detection based on color variance
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_variance = np.var(hsv[:, :, 0])  # Hue variance
        
        # High color variance might indicate obstruction
        return color_variance > 1000
    
    def _analyze_overall_problems(self, image: np.ndarray, detections: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze overall image for problems"""
        
        problems = []
        
        # Check for missing expected elements
        detected_classes = [d['class_name'] for d in detections]
        
        # Look for missing critical elements
        if 'zebra_crossing' in detected_classes and 'pedestrian_warning_sign' not in detected_classes:
            problems.append({
                'type': 'missing',
                'object': 'pedestrian_warning_sign',
                'confidence': 0.7,
                'description': 'Pedestrian crossing detected but warning sign missing',
                'severity': 'high',
                'location': 'nearby'
            })
        
        if 'school_zone' in detected_classes and 'speed_limit' not in detected_classes:
            problems.append({
                'type': 'missing',
                'object': 'speed_limit_sign',
                'confidence': 0.8,
                'description': 'School zone detected but speed limit sign missing',
                'severity': 'high',
                'location': 'approach'
            })
        
        return problems
    
    def _generate_vision_recommendations(self, analysis_result: Dict) -> List[Dict[str, Any]]:
        """Generate recommendations based on vision analysis"""
        
        recommendations = []
        problems = analysis_result.get('problems_detected', [])
        
        for problem in problems:
            if problem['type'] == 'faded':
                recommendations.append({
                    'intervention': 'Repaint faded markings/signs',
                    'priority': 'medium',
                    'estimated_cost': 15000,
                    'timeline': '2 days',
                    'expected_impact': '30% visibility improvement'
                })
            
            elif problem['type'] == 'damaged':
                recommendations.append({
                    'intervention': 'Replace damaged infrastructure',
                    'priority': 'high',
                    'estimated_cost': 25000,
                    'timeline': '1 week',
                    'expected_impact': '50% safety improvement'
                })
            
            elif problem['type'] == 'missing':
                recommendations.append({
                    'intervention': 'Install missing safety elements',
                    'priority': 'high',
                    'estimated_cost': 20000,
                    'timeline': '3 days',
                    'expected_impact': '40% accident reduction'
                })
            
            elif problem['type'] == 'obstructed':
                recommendations.append({
                    'intervention': 'Clear obstructions and improve visibility',
                    'priority': 'high',
                    'estimated_cost': 10000,
                    'timeline': '1 day',
                    'expected_impact': '25% visibility improvement'
                })
        
        return recommendations
    
    def _fallback_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback analysis when model is not available"""
        
        return {
            'detections': [],
            'total_detections': 0,
            'average_confidence': 0,
            'image_dimensions': [0, 0],
            'analysis_timestamp': self._get_timestamp(),
            'problems_detected': [
                {
                    'type': 'analysis_unavailable',
                    'object': 'unknown',
                    'confidence': 0.5,
                    'description': 'Computer vision analysis not available. Please provide manual description.',
                    'severity': 'low',
                    'location': 'entire_image'
                }
            ],
            'recommendations': [
                {
                    'intervention': 'Manual assessment required',
                    'priority': 'medium',
                    'estimated_cost': 0,
                    'timeline': 'immediate',
                    'expected_impact': 'unknown'
                }
            ],
            'fallback_mode': True
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_analyze_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple images in batch"""
        
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                results.append(self._fallback_analysis(image_path))
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'model_loaded': self.model is not None,
            'road_safety_classes': len(self.road_safety_classes),
            'problem_patterns': len(self.problem_patterns)
        }

class VisionIntegrationEngine:
    """Integration engine for combining vision analysis with intervention recommendations"""
    
    def __init__(self, vision_analyzer: RoadSafetyVisionAnalyzer):
        self.vision_analyzer = vision_analyzer
    
    def process_image_with_recommendations(self, image_path: str, 
                                         intervention_database: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process image and generate intervention recommendations"""
        
        # Analyze image
        vision_result = self.vision_analyzer.analyze_image(image_path)
        
        # Find matching interventions
        matching_interventions = self._find_matching_interventions(
            vision_result, intervention_database
        )
        
        # Generate scenarios
        scenarios = self._generate_vision_scenarios(vision_result, matching_interventions)
        
        return {
            'vision_analysis': vision_result,
            'matching_interventions': matching_interventions,
            'recommended_scenarios': scenarios,
            'integration_timestamp': self._get_timestamp()
        }
    
    def _find_matching_interventions(self, vision_result: Dict, 
                                   intervention_database: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find interventions matching vision analysis results"""
        
        matching_interventions = []
        problems = vision_result.get('problems_detected', [])
        
        for problem in problems:
            problem_type = problem['type']
            object_type = problem['object']
            
            # Find interventions matching the problem
            for intervention in intervention_database:
                if (intervention['problem_type'] == problem_type and 
                    object_type in intervention['intervention_name'].lower()):
                    
                    # Add vision-specific metadata
                    intervention_copy = intervention.copy()
                    intervention_copy['vision_detected'] = True
                    intervention_copy['vision_confidence'] = problem['confidence']
                    intervention_copy['vision_location'] = problem['location']
                    
                    matching_interventions.append(intervention_copy)
        
        return matching_interventions
    
    def _generate_vision_scenarios(self, vision_result: Dict, 
                                 interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate intervention scenarios based on vision analysis"""
        
        scenarios = []
        
        if not interventions:
            return scenarios
        
        # Quick fix scenario
        quick_fix_interventions = [i for i in interventions if i['implementation_timeline'] <= 2]
        if quick_fix_interventions:
            scenarios.append({
                'scenario_name': 'Quick Fix',
                'interventions': quick_fix_interventions[:3],
                'total_cost': sum(i['cost_estimate']['total'] for i in quick_fix_interventions[:3]),
                'timeline': max(i['implementation_timeline'] for i in quick_fix_interventions[:3]),
                'expected_impact': sum(i['predicted_impact']['accident_reduction_percent'] for i in quick_fix_interventions[:3]) / len(quick_fix_interventions[:3]),
                'description': 'Immediate repairs and basic improvements'
            })
        
        # Comprehensive scenario
        all_interventions = interventions[:5]  # Top 5 interventions
        scenarios.append({
            'scenario_name': 'Comprehensive Solution',
            'interventions': all_interventions,
            'total_cost': sum(i['cost_estimate']['total'] for i in all_interventions),
            'timeline': max(i['implementation_timeline'] for i in all_interventions),
            'expected_impact': sum(i['predicted_impact']['accident_reduction_percent'] for i in all_interventions) / len(all_interventions),
            'description': 'Complete safety improvement package'
        })
        
        return scenarios
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

def initialize_vision_analyzer() -> RoadSafetyVisionAnalyzer:
    """Initialize the vision analyzer"""
    try:
        analyzer = RoadSafetyVisionAnalyzer()
        return analyzer
    except Exception as e:
        logger.error(f"Failed to initialize vision analyzer: {e}")
        return None
