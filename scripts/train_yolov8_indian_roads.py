#!/usr/bin/env python3
"""
YOLOv8 Training Script for Indian Road Infrastructure
Trains custom YOLOv8 model on Indian road dataset with data augmentation
"""

import os
import sys
import logging
import random
import json
import cv2
import numpy as np
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime
import yaml

# YOLOv8 imports
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

logger = logging.getLogger(__name__)

class IndianRoadDatasetGenerator:
    """Generate synthetic Indian road dataset for YOLOv8 training"""
    
    def __init__(self):
        self.output_dir = Path("data/vision/dataset")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Indian road infrastructure classes
        self.classes = {
            0: "traffic_sign",
            1: "zebra_crossing", 
            2: "speed_bump",
            3: "guard_rail",
            4: "street_light",
            5: "pedestrian_sign",
            6: "stop_sign",
            7: "speed_limit_sign",
            8: "warning_sign",
            9: "traffic_signal",
            10: "lane_marking",
            11: "road_barrier",
            12: "pedestrian_bridge",
            13: "cycle_lane",
            14: "bus_stop"
        }
        
        # Indian road characteristics
        self.road_colors = {
            "asphalt": (64, 64, 64),
            "concrete": (128, 128, 128),
            "dirt": (139, 69, 19),
            "gravel": (105, 105, 105)
        }
        
        self.sign_colors = {
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        
        # Indian traffic sign templates
        self.sign_templates = {
            "stop_sign": {"shape": "octagon", "color": "red", "text": "STOP"},
            "speed_limit_sign": {"shape": "circle", "color": "white", "text": "30"},
            "warning_sign": {"shape": "triangle", "color": "yellow", "text": "!"},
            "pedestrian_sign": {"shape": "square", "color": "blue", "text": "PED"},
            "traffic_signal": {"shape": "circle", "color": "red", "text": "●"}
        }
    
    def generate_synthetic_image(self, width: int = 640, height: int = 640) -> Tuple[np.ndarray, List[Dict]]:
        """Generate synthetic Indian road image with annotations"""
        
        # Create base image
        image = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add road surface
        road_color = random.choice(list(self.road_colors.values()))
        road_height = random.randint(height // 3, height // 2)
        road_y = height - road_height
        image[road_y:, :] = road_color
        
        # Add lane markings
        self._add_lane_markings(image, road_y, width, height)
        
        annotations = []
        
        # Add random road infrastructure
        num_objects = random.randint(3, 8)
        for _ in range(num_objects):
            obj_type = random.choice(list(self.classes.keys()))
            obj_name = self.classes[obj_type]
            
            if obj_name in ["traffic_sign", "stop_sign", "speed_limit_sign", "warning_sign", "pedestrian_sign"]:
                bbox, annotation = self._add_traffic_sign(image, obj_name, width, height)
            elif obj_name == "zebra_crossing":
                bbox, annotation = self._add_zebra_crossing(image, road_y, width, height)
            elif obj_name == "speed_bump":
                bbox, annotation = self._add_speed_bump(image, road_y, width, height)
            elif obj_name == "guard_rail":
                bbox, annotation = self._add_guard_rail(image, road_y, width, height)
            elif obj_name == "street_light":
                bbox, annotation = self._add_street_light(image, road_y, width, height)
            elif obj_name == "traffic_signal":
                bbox, annotation = self._add_traffic_signal(image, road_y, width, height)
            elif obj_name == "lane_marking":
                bbox, annotation = self._add_lane_marking(image, road_y, width, height)
            elif obj_name == "road_barrier":
                bbox, annotation = self._add_road_barrier(image, road_y, width, height)
            elif obj_name == "pedestrian_bridge":
                bbox, annotation = self._add_pedestrian_bridge(image, road_y, width, height)
            elif obj_name == "cycle_lane":
                bbox, annotation = self._add_cycle_lane(image, road_y, width, height)
            elif obj_name == "bus_stop":
                bbox, annotation = self._add_bus_stop(image, road_y, width, height)
            else:
                continue
            
            if bbox is not None:
                annotations.append(annotation)
        
        # Add weather effects
        self._add_weather_effects(image)
        
        # Add noise
        self._add_noise(image)
        
        return image, annotations
    
    def _add_lane_markings(self, image: np.ndarray, road_y: int, width: int, height: int):
        """Add lane markings to road"""
        # Center line
        center_x = width // 2
        for y in range(road_y, height, 20):
            cv2.line(image, (center_x - 50, y), (center_x + 50, y), (255, 255, 255), 3)
        
        # Side lines
        cv2.line(image, (50, road_y), (50, height), (255, 255, 255), 2)
        cv2.line(image, (width - 50, road_y), (width - 50, height), (255, 255, 255), 2)
    
    def _add_traffic_sign(self, image: np.ndarray, sign_type: str, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add traffic sign to image"""
        sign_size = random.randint(30, 80)
        x = random.randint(50, width - sign_size - 50)
        y = random.randint(50, height - sign_size - 50)
        
        # Get sign template
        template = self.sign_templates.get(sign_type, {"shape": "circle", "color": "red", "text": "SIGN"})
        
        # Draw sign based on shape
        if template["shape"] == "circle":
            cv2.circle(image, (x + sign_size//2, y + sign_size//2), sign_size//2, self.sign_colors[template["color"]], -1)
            cv2.circle(image, (x + sign_size//2, y + sign_size//2), sign_size//2, (0, 0, 0), 2)
        elif template["shape"] == "triangle":
            pts = np.array([[x + sign_size//2, y], [x, y + sign_size], [x + sign_size, y + sign_size]], np.int32)
            cv2.fillPoly(image, [pts], self.sign_colors[template["color"]])
            cv2.polylines(image, [pts], True, (0, 0, 0), 2)
        elif template["shape"] == "octagon":
            pts = np.array([[x + sign_size//4, y], [x + 3*sign_size//4, y], 
                           [x + sign_size, y + sign_size//4], [x + sign_size, y + 3*sign_size//4],
                           [x + 3*sign_size//4, y + sign_size], [x + sign_size//4, y + sign_size],
                           [x, y + 3*sign_size//4], [x, y + sign_size//4]], np.int32)
            cv2.fillPoly(image, [pts], self.sign_colors[template["color"]])
            cv2.polylines(image, [pts], True, (0, 0, 0), 2)
        else:  # square
            cv2.rectangle(image, (x, y), (x + sign_size, y + sign_size), self.sign_colors[template["color"]], -1)
            cv2.rectangle(image, (x, y), (x + sign_size, y + sign_size), (0, 0, 0), 2)
        
        # Add text
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(template["text"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x + (sign_size - text_size[0]) // 2
        text_y = y + (sign_size + text_size[1]) // 2
        cv2.putText(image, template["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Convert to YOLO format (normalized)
        bbox = (x / width, y / height, sign_size / width, sign_size / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index(sign_type)]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": sign_type
        }
        
        return bbox, annotation
    
    def _add_zebra_crossing(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add zebra crossing to road"""
        crossing_width = random.randint(80, 150)
        crossing_height = random.randint(20, 40)
        x = random.randint(50, width - crossing_width - 50)
        y = road_y + random.randint(0, height - road_y - crossing_height)
        
        # Draw zebra stripes
        stripe_width = crossing_width // 8
        for i in range(8):
            stripe_x = x + i * stripe_width
            if i % 2 == 0:  # White stripe
                cv2.rectangle(image, (stripe_x, y), (stripe_x + stripe_width, y + crossing_height), (255, 255, 255), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, crossing_width / width, crossing_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("zebra_crossing")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "zebra_crossing"
        }
        
        return bbox, annotation
    
    def _add_speed_bump(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add speed bump to road"""
        bump_width = random.randint(100, 200)
        bump_height = random.randint(15, 30)
        x = random.randint(50, width - bump_width - 50)
        y = road_y + random.randint(0, height - road_y - bump_height)
        
        # Draw speed bump
        cv2.rectangle(image, (x, y), (x + bump_width, y + bump_height), (139, 69, 19), -1)
        cv2.rectangle(image, (x, y), (x + bump_width, y + bump_height), (0, 0, 0), 2)
        
        # Add reflective strips
        for i in range(3):
            strip_x = x + i * (bump_width // 3)
            cv2.rectangle(image, (strip_x, y), (strip_x + bump_width//3, y + bump_height), (255, 255, 0), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, bump_width / width, bump_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("speed_bump")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "speed_bump"
        }
        
        return bbox, annotation
    
    def _add_guard_rail(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add guard rail to road"""
        rail_width = random.randint(200, 400)
        rail_height = random.randint(20, 40)
        x = random.randint(50, width - rail_width - 50)
        y = road_y + random.randint(0, height - road_y - rail_height)
        
        # Draw guard rail
        cv2.rectangle(image, (x, y), (x + rail_width, y + rail_height), (105, 105, 105), -1)
        cv2.rectangle(image, (x, y), (x + rail_width, y + rail_height), (0, 0, 0), 2)
        
        # Add posts
        post_spacing = rail_width // 5
        for i in range(6):
            post_x = x + i * post_spacing
            cv2.rectangle(image, (post_x, y), (post_x + 5, y + rail_height), (0, 0, 0), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, rail_width / width, rail_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("guard_rail")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "guard_rail"
        }
        
        return bbox, annotation
    
    def _add_street_light(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add street light to image"""
        light_width = random.randint(15, 30)
        light_height = random.randint(60, 120)
        x = random.randint(50, width - light_width - 50)
        y = random.randint(50, height - light_height - 50)
        
        # Draw pole
        cv2.rectangle(image, (x + light_width//2 - 3, y), (x + light_width//2 + 3, y + light_height), (105, 105, 105), -1)
        
        # Draw light
        cv2.circle(image, (x + light_width//2, y), light_width//2, (255, 255, 200), -1)
        cv2.circle(image, (x + light_width//2, y), light_width//2, (0, 0, 0), 2)
        
        # Add light glow effect
        cv2.circle(image, (x + light_width//2, y), light_width//2 + 5, (255, 255, 200), 1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, light_width / width, light_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("street_light")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "street_light"
        }
        
        return bbox, annotation
    
    def _add_traffic_signal(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add traffic signal to image"""
        signal_width = random.randint(40, 80)
        signal_height = random.randint(100, 150)
        x = random.randint(50, width - signal_width - 50)
        y = random.randint(50, height - signal_height - 50)
        
        # Draw signal box
        cv2.rectangle(image, (x, y), (x + signal_width, y + signal_height), (0, 0, 0), -1)
        cv2.rectangle(image, (x, y), (x + signal_width, y + signal_height), (255, 255, 255), 2)
        
        # Draw lights
        light_size = signal_width // 3
        light_y_spacing = signal_height // 4
        
        # Red light
        cv2.circle(image, (x + signal_width//2, y + light_y_spacing), light_size//2, (0, 0, 255), -1)
        cv2.circle(image, (x + signal_width//2, y + light_y_spacing), light_size//2, (255, 255, 255), 2)
        
        # Yellow light
        cv2.circle(image, (x + signal_width//2, y + 2*light_y_spacing), light_size//2, (0, 255, 255), -1)
        cv2.circle(image, (x + signal_width//2, y + 2*light_y_spacing), light_size//2, (255, 255, 255), 2)
        
        # Green light
        cv2.circle(image, (x + signal_width//2, y + 3*light_y_spacing), light_size//2, (0, 255, 0), -1)
        cv2.circle(image, (x + signal_width//2, y + 3*light_y_spacing), light_size//2, (255, 255, 255), 2)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, signal_width / width, signal_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("traffic_signal")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "traffic_signal"
        }
        
        return bbox, annotation
    
    def _add_lane_marking(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add lane marking to road"""
        marking_width = random.randint(50, 150)
        marking_height = random.randint(5, 15)
        x = random.randint(50, width - marking_width - 50)
        y = road_y + random.randint(0, height - road_y - marking_height)
        
        # Draw lane marking
        cv2.rectangle(image, (x, y), (x + marking_width, y + marking_height), (255, 255, 255), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, marking_width / width, marking_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("lane_marking")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "lane_marking"
        }
        
        return bbox, annotation
    
    def _add_road_barrier(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add road barrier to image"""
        barrier_width = random.randint(100, 300)
        barrier_height = random.randint(30, 60)
        x = random.randint(50, width - barrier_width - 50)
        y = random.randint(50, height - barrier_height - 50)
        
        # Draw barrier
        cv2.rectangle(image, (x, y), (x + barrier_width, y + barrier_height), (128, 128, 128), -1)
        cv2.rectangle(image, (x, y), (x + barrier_width, y + barrier_height), (0, 0, 0), 2)
        
        # Add reflective strips
        for i in range(3):
            strip_y = y + i * (barrier_height // 3)
            cv2.rectangle(image, (x, strip_y), (x + barrier_width, strip_y + 5), (255, 255, 0), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, barrier_width / width, barrier_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("road_barrier")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "road_barrier"
        }
        
        return bbox, annotation
    
    def _add_pedestrian_bridge(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add pedestrian bridge to image"""
        bridge_width = random.randint(150, 300)
        bridge_height = random.randint(80, 150)
        x = random.randint(50, width - bridge_width - 50)
        y = random.randint(50, height - bridge_height - 50)
        
        # Draw bridge deck
        cv2.rectangle(image, (x, y), (x + bridge_width, y + bridge_height), (139, 69, 19), -1)
        cv2.rectangle(image, (x, y), (x + bridge_width, y + bridge_height), (0, 0, 0), 2)
        
        # Draw supports
        support_spacing = bridge_width // 3
        for i in range(4):
            support_x = x + i * support_spacing
            cv2.rectangle(image, (support_x, y + bridge_height), (support_x + 10, y + bridge_height + 20), (105, 105, 105), -1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, bridge_width / width, bridge_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("pedestrian_bridge")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "pedestrian_bridge"
        }
        
        return bbox, annotation
    
    def _add_cycle_lane(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add cycle lane to road"""
        lane_width = random.randint(80, 150)
        lane_height = random.randint(20, 40)
        x = random.randint(50, width - lane_width - 50)
        y = road_y + random.randint(0, height - road_y - lane_height)
        
        # Draw cycle lane
        cv2.rectangle(image, (x, y), (x + lane_width, y + lane_height), (0, 255, 0), -1)
        
        # Add cycle symbols
        symbol_spacing = lane_width // 4
        for i in range(3):
            symbol_x = x + i * symbol_spacing + symbol_spacing//2
            symbol_y = y + lane_height//2
            cv2.circle(image, (symbol_x, symbol_y), 8, (255, 255, 255), -1)
            cv2.circle(image, (symbol_x, symbol_y), 8, (0, 0, 0), 2)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, lane_width / width, lane_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("cycle_lane")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "cycle_lane"
        }
        
        return bbox, annotation
    
    def _add_bus_stop(self, image: np.ndarray, road_y: int, width: int, height: int) -> Tuple[Tuple, Dict]:
        """Add bus stop to image"""
        stop_width = random.randint(60, 120)
        stop_height = random.randint(80, 150)
        x = random.randint(50, width - stop_width - 50)
        y = random.randint(50, height - stop_height - 50)
        
        # Draw bus stop structure
        cv2.rectangle(image, (x, y), (x + stop_width, y + stop_height), (135, 206, 235), -1)
        cv2.rectangle(image, (x, y), (x + stop_width, y + stop_height), (0, 0, 0), 2)
        
        # Add roof
        cv2.rectangle(image, (x - 10, y - 20), (x + stop_width + 10, y), (105, 105, 105), -1)
        
        # Add bench
        cv2.rectangle(image, (x + 10, y + stop_height - 30), (x + stop_width - 10, y + stop_height - 20), (139, 69, 19), -1)
        
        # Add bus stop sign
        cv2.rectangle(image, (x + stop_width//2 - 10, y + 20), (x + stop_width//2 + 10, y + 40), (255, 255, 0), -1)
        cv2.putText(image, "BUS", (x + stop_width//2 - 8, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Convert to YOLO format
        bbox = (x / width, y / height, stop_width / width, stop_height / height)
        
        class_id = list(self.classes.keys())[list(self.classes.values()).index("bus_stop")]
        annotation = {
            "class_id": class_id,
            "bbox": bbox,
            "class_name": "bus_stop"
        }
        
        return bbox, annotation
    
    def _add_weather_effects(self, image: np.ndarray):
        """Add weather effects to image"""
        weather_type = random.choice(["clear", "rain", "fog", "dust"])
        
        if weather_type == "rain":
            # Add rain streaks
            for _ in range(100):
                x = random.randint(0, image.shape[1])
                y = random.randint(0, image.shape[0])
                cv2.line(image, (x, y), (x + 2, y + 10), (200, 200, 200), 1)
        
        elif weather_type == "fog":
            # Add fog effect
            fog_overlay = np.ones_like(image) * 200
            alpha = random.uniform(0.1, 0.3)
            image = cv2.addWeighted(image, 1 - alpha, fog_overlay, alpha, 0)
        
        elif weather_type == "dust":
            # Add dust particles
            for _ in range(50):
                x = random.randint(0, image.shape[1])
                y = random.randint(0, image.shape[0])
                cv2.circle(image, (x, y), 1, (139, 69, 19), -1)
    
    def _add_noise(self, image: np.ndarray):
        """Add noise to image"""
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
    
    def generate_dataset(self, num_images: int = 1000):
        """Generate complete dataset"""
        logger.info(f"Generating {num_images} synthetic images...")
        
        # Create directories
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        # Split dataset
        train_split = int(num_images * 0.7)
        val_split = int(num_images * 0.2)
        test_split = num_images - train_split - val_split
        
        splits = {
            "train": train_split,
            "val": val_split,
            "test": test_split
        }
        
        image_count = 0
        
        for split_name, split_count in splits.items():
            logger.info(f"Generating {split_count} images for {split_name} split...")
            
            split_images_dir = images_dir / split_name
            split_labels_dir = labels_dir / split_name
            split_images_dir.mkdir(exist_ok=True)
            split_labels_dir.mkdir(exist_ok=True)
            
            for i in range(split_count):
                # Generate image and annotations
                image, annotations = self.generate_synthetic_image()
                
                # Save image
                image_filename = f"{split_name}_{i:06d}.jpg"
                image_path = split_images_dir / image_filename
                cv2.imwrite(str(image_path), image)
                
                # Save labels in YOLO format
                label_filename = f"{split_name}_{i:06d}.txt"
                label_path = split_labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for annotation in annotations:
                        class_id = annotation["class_id"]
                        bbox = annotation["bbox"]
                        f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
                image_count += 1
                
                if image_count % 100 == 0:
                    logger.info(f"Generated {image_count}/{num_images} images...")
        
        # Create dataset configuration
        self._create_dataset_config()
        
        logger.info(f"Dataset generation complete: {num_images} images")
    
    def _create_dataset_config(self):
        """Create dataset configuration file"""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": list(self.classes.values())
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved to {config_path}")

class YOLOv8IndianRoadTrainer:
    """Train YOLOv8 model on Indian road dataset"""
    
    def __init__(self):
        self.model = None
        self.dataset_path = Path("data/vision/dataset")
        self.model_path = Path("models/vision")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.training_params = {
            "epochs": 50,
            "batch_size": 16,
            "imgsz": 640,
            "device": "cpu",  # Change to "cuda" if GPU available
            "workers": 4,
            "patience": 10,
            "save_period": 10,
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 2.0,
            "label_smoothing": 0.0,
            "nbs": 64,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True
        }
    
    def setup_model(self):
        """Setup YOLOv8 model"""
        try:
            logger.info("Setting up YOLOv8 model...")
            
            # Load YOLOv8n model (nano version for faster training)
            self.model = YOLO('yolov8n.pt')
            
            logger.info("YOLOv8 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up YOLOv8 model: {e}")
            return False
    
    def train_model(self):
        """Train YOLOv8 model on Indian road dataset"""
        try:
            if not self.model:
                logger.error("Model not setup. Call setup_model() first.")
                return False
            
            logger.info("Starting YOLOv8 training...")
            
            # Check if dataset exists
            dataset_config = self.dataset_path / "dataset.yaml"
            if not dataset_config.exists():
                logger.error(f"Dataset configuration not found at {dataset_config}")
                return False
            
            # Train the model
            results = self.model.train(
                data=str(dataset_config),
                epochs=self.training_params["epochs"],
                batch=self.training_params["batch_size"],
                imgsz=self.training_params["imgsz"],
                device=self.training_params["device"],
                workers=self.training_params["workers"],
                patience=self.training_params["patience"],
                save_period=self.training_params["save_period"],
                lr0=self.training_params["lr0"],
                lrf=self.training_params["lrf"],
                momentum=self.training_params["momentum"],
                weight_decay=self.training_params["weight_decay"],
                warmup_epochs=self.training_params["warmup_epochs"],
                warmup_momentum=self.training_params["warmup_momentum"],
                warmup_bias_lr=self.training_params["warmup_bias_lr"],
                box=self.training_params["box"],
                cls=self.training_params["cls"],
                dfl=self.training_params["dfl"],
                pose=self.training_params["pose"],
                kobj=self.training_params["kobj"],
                label_smoothing=self.training_params["label_smoothing"],
                nbs=self.training_params["nbs"],
                overlap_mask=self.training_params["overlap_mask"],
                mask_ratio=self.training_params["mask_ratio"],
                dropout=self.training_params["dropout"],
                val=self.training_params["val"],
                project=str(self.model_path),
                name="yolov8_indian_roads"
            )
            
            logger.info("YOLOv8 training completed successfully!")
            
            # Save best model
            best_model_path = self.model_path / "yolov8_indian_roads" / "weights" / "best.pt"
            if best_model_path.exists():
                # Copy to main models directory
                import shutil
                final_model_path = self.model_path / "yolov8_indian_roads.pt"
                shutil.copy2(best_model_path, final_model_path)
                logger.info(f"Best model saved to {final_model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training YOLOv8 model: {e}")
            return False
    
    def validate_model(self):
        """Validate trained model"""
        try:
            logger.info("Validating trained model...")
            
            # Load best model
            model_path = self.model_path / "yolov8_indian_roads.pt"
            if not model_path.exists():
                logger.error(f"Trained model not found at {model_path}")
                return False
            
            # Load model
            model = YOLO(str(model_path))
            
            # Validate on test set
            dataset_config = self.dataset_path / "dataset.yaml"
            results = model.val(data=str(dataset_config))
            
            logger.info("Model validation completed!")
            logger.info(f"mAP50: {results.box.map50:.3f}")
            logger.info(f"mAP50-95: {results.box.map:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def test_inference(self):
        """Test model inference on sample images"""
        try:
            logger.info("Testing model inference...")
            
            # Load best model
            model_path = self.model_path / "yolov8_indian_roads.pt"
            if not model_path.exists():
                logger.error(f"Trained model not found at {model_path}")
                return False
            
            # Load model
            model = YOLO(str(model_path))
            
            # Test on sample images
            test_images_dir = self.dataset_path / "images" / "test"
            if test_images_dir.exists():
                test_images = list(test_images_dir.glob("*.jpg"))[:5]  # Test first 5 images
                
                for image_path in test_images:
                    # Run inference
                    results = model(str(image_path))
                    
                    # Save results
                    output_path = self.model_path / f"inference_{image_path.name}"
                    results[0].save(str(output_path))
                    
                    logger.info(f"Inference result saved to {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing inference: {e}")
            return False

async def main():
    """Main function to train YOLOv8 on Indian roads"""
    logging.basicConfig(level=logging.INFO)
    
    # Generate dataset
    logger.info("Generating Indian road dataset...")
    dataset_generator = IndianRoadDatasetGenerator()
    dataset_generator.generate_dataset(num_images=1000)
    
    # Train model
    logger.info("Training YOLOv8 model...")
    trainer = YOLOv8IndianRoadTrainer()
    
    if trainer.setup_model():
        if trainer.train_model():
            trainer.validate_model()
            trainer.test_inference()
            print("YOLOv8 training completed successfully!")
        else:
            print("YOLOv8 training failed!")
    else:
        print("YOLOv8 model setup failed!")

if __name__ == "__main__":
    asyncio.run(main())
