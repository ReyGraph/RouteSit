"""
Continuous Learning System for Routesit AI
Self-learning pipeline with user feedback and incremental updates
Grows smarter with every user interaction
"""

import os
import json
import logging
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import pickle
from pathlib import Path
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """User feedback on system recommendations"""
    feedback_id: str
    query_id: str
    user_id: Optional[str]
    recommendation_id: str
    feedback_type: str  # "accepted", "rejected", "modified", "rating"
    feedback_value: Union[bool, int, str]
    timestamp: str
    context: Dict[str, Any]
    confidence: float

@dataclass
class LearningExample:
    """Training example for continuous learning"""
    example_id: str
    input_features: Dict[str, Any]
    ground_truth: Any
    prediction: Any
    confidence: float
    source: str  # "user_feedback", "expert_validation", "synthetic"
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_threshold: float
    evaluation_timestamp: str
    test_set_size: int

class ContinuousLearner:
    """
    Self-learning system that improves with user feedback
    Implements active learning, experience replay, and incremental updates
    """
    
    def __init__(self):
        self.feedback_buffer = deque(maxlen=10000)  # Circular buffer for feedback
        self.training_examples = deque(maxlen=50000)  # Training examples
        self.models = {}  # Different models for different tasks
        self.performance_history = []
        
        # Learning parameters
        self.learning_threshold = 100  # Trigger learning after N examples
        self.confidence_threshold = 0.7  # Minimum confidence for predictions
        self.active_learning_threshold = 0.5  # Uncertainty threshold for active learning
        
        # Data storage paths
        self.models_path = Path("models/continuous_learning")
        self.data_path = Path("data/learning")
        self.feedback_path = Path("data/feedback")
        
        # Create directories
        for path in [self.models_path, self.data_path, self.feedback_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe queues for async processing
        self.feedback_queue = queue.Queue()
        self.learning_queue = queue.Queue()
        
        # Background processing thread
        self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.processing_thread.start()
        
        # Load existing models and data
        self._load_existing_data()
        
        logger.info("Continuous learning system initialized")
    
    def add_user_feedback(self, feedback: UserFeedback):
        """Add user feedback to learning system"""
        try:
            # Add to feedback buffer
            self.feedback_buffer.append(feedback)
            
            # Add to processing queue
            self.feedback_queue.put(feedback)
            
            # Save to disk
            self._save_feedback(feedback)
            
            logger.info(f"Added feedback: {feedback.feedback_type} for recommendation {feedback.recommendation_id}")
            
        except Exception as e:
            logger.error(f"Error adding user feedback: {e}")
    
    def process_user_query(self, query: Dict[str, Any], prediction: Any, confidence: float) -> Dict[str, Any]:
        """
        Process user query and generate prediction
        Main entry point for the learning system
        """
        try:
            # Generate prediction
            enhanced_prediction = self._enhance_prediction(query, prediction, confidence)
            
            # Check if we need active learning
            if confidence < self.active_learning_threshold:
                enhanced_prediction['needs_human_review'] = True
                enhanced_prediction['uncertainty_reason'] = "Low confidence prediction"
            
            # Create learning example
            learning_example = LearningExample(
                example_id=str(uuid.uuid4()),
                input_features=self._extract_features(query),
                ground_truth=None,  # Will be filled when feedback received
                prediction=enhanced_prediction,
                confidence=confidence,
                source="user_query",
                timestamp=datetime.now().isoformat(),
                metadata={"query_type": query.get('type', 'unknown')}
            )
            
            # Add to training examples
            self.training_examples.append(learning_example)
            
            # Check if we need to trigger learning
            if len(self.training_examples) >= self.learning_threshold:
                self._trigger_incremental_learning()
            
            return enhanced_prediction
            
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return prediction
    
    def _enhance_prediction(self, query: Dict[str, Any], prediction: Any, confidence: float) -> Dict[str, Any]:
        """Enhance prediction with learned patterns"""
        try:
            enhanced = {
                'prediction': prediction,
                'confidence': confidence,
                'enhancements': []
            }
            
            # Apply learned patterns
            if confidence > self.confidence_threshold:
                # High confidence - apply learned improvements
                enhanced['enhancements'].append("High confidence prediction")
                
                # Add learned context
                similar_cases = self._find_similar_cases(query)
                if similar_cases:
                    enhanced['similar_cases'] = similar_cases
                    enhanced['enhancements'].append("Similar case analysis")
            
            else:
                # Low confidence - flag for review
                enhanced['needs_human_review'] = True
                enhanced['enhancements'].append("Low confidence - needs review")
            
            # Add learned recommendations
            learned_recommendations = self._get_learned_recommendations(query)
            if learned_recommendations:
                enhanced['learned_recommendations'] = learned_recommendations
                enhanced['enhancements'].append("Learned recommendations")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prediction: {e}")
            return {'prediction': prediction, 'confidence': confidence}
    
    def _extract_features(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from user query for learning"""
        features = {
            'text_length': len(str(query.get('text', ''))),
            'has_image': 'image' in query and query['image'] is not None,
            'has_location': 'location' in query and query['location'] is not None,
            'query_type': query.get('type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract intervention-related features
        if 'interventions' in query:
            features['intervention_count'] = len(query['interventions'])
            features['intervention_types'] = query['interventions']
        
        # Extract location features
        if 'location' in query:
            location = query['location']
            features['city'] = location.get('city', 'unknown')
            features['state'] = location.get('state', 'unknown')
            features['road_type'] = location.get('road_type', 'unknown')
        
        return features
    
    def _find_similar_cases(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar cases from training examples"""
        try:
            query_features = self._extract_features(query)
            similar_cases = []
            
            # Simple similarity based on feature overlap
            for example in list(self.training_examples)[-1000:]:  # Last 1000 examples
                if example.ground_truth is None:
                    continue
                
                similarity = self._calculate_similarity(query_features, example.input_features)
                if similarity > 0.7:  # High similarity threshold
                    similar_cases.append({
                        'example_id': example.example_id,
                        'similarity': similarity,
                        'prediction': example.prediction,
                        'ground_truth': example.ground_truth,
                        'confidence': example.confidence
                    })
            
            # Sort by similarity and return top 5
            similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_cases[:5]
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets"""
        try:
            # Simple Jaccard similarity for categorical features
            categorical_features = ['city', 'state', 'road_type', 'query_type']
            
            matches = 0
            total = 0
            
            for feature in categorical_features:
                if feature in features1 and feature in features2:
                    total += 1
                    if features1[feature] == features2[feature]:
                        matches += 1
            
            if total == 0:
                return 0.0
            
            # Add numerical feature similarity
            numerical_features = ['text_length', 'intervention_count']
            for feature in numerical_features:
                if feature in features1 and feature in features2:
                    total += 1
                    # Normalize difference
                    diff = abs(features1[feature] - features2[feature])
                    max_val = max(features1[feature], features2[feature])
                    if max_val > 0:
                        similarity = 1 - (diff / max_val)
                        matches += similarity
            
            return matches / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _get_learned_recommendations(self, query: Dict[str, Any]) -> List[str]:
        """Get learned recommendations based on similar cases"""
        try:
            similar_cases = self._find_similar_cases(query)
            recommendations = []
            
            for case in similar_cases:
                if case['similarity'] > 0.8:  # Very high similarity
                    if isinstance(case['ground_truth'], list):
                        recommendations.extend(case['ground_truth'])
                    else:
                        recommendations.append(str(case['ground_truth']))
            
            # Remove duplicates and return unique recommendations
            return list(set(recommendations))
            
        except Exception as e:
            logger.error(f"Error getting learned recommendations: {e}")
            return []
    
    def _trigger_incremental_learning(self):
        """Trigger incremental model updates"""
        try:
            logger.info("Triggering incremental learning...")
            
            # Add to learning queue for background processing
            self.learning_queue.put("incremental_update")
            
        except Exception as e:
            logger.error(f"Error triggering incremental learning: {e}")
    
    def _background_processor(self):
        """Background thread for processing feedback and learning"""
        while True:
            try:
                # Process feedback
                if not self.feedback_queue.empty():
                    feedback = self.feedback_queue.get_nowait()
                    self._process_feedback(feedback)
                
                # Process learning
                if not self.learning_queue.empty():
                    learning_task = self.learning_queue.get_nowait()
                    if learning_task == "incremental_update":
                        self._perform_incremental_update()
                
                # Sleep briefly to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                time.sleep(1)
    
    def _process_feedback(self, feedback: UserFeedback):
        """Process individual feedback item"""
        try:
            # Find corresponding training example
            for example in self.training_examples:
                if example.example_id == feedback.query_id:
                    # Update ground truth based on feedback
                    if feedback.feedback_type == "accepted":
                        example.ground_truth = example.prediction
                    elif feedback.feedback_type == "rejected":
                        example.ground_truth = "rejected"
                    elif feedback.feedback_type == "modified":
                        example.ground_truth = feedback.feedback_value
                    
                    example.metadata['feedback_received'] = True
                    example.metadata['feedback_type'] = feedback.feedback_type
                    break
            
            logger.info(f"Processed feedback: {feedback.feedback_type}")
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
    
    def _perform_incremental_update(self):
        """Perform incremental model update"""
        try:
            logger.info("Performing incremental model update...")
            
            # Get recent training examples with ground truth
            training_data = []
            for example in self.training_examples:
                if example.ground_truth is not None:
                    training_data.append(example)
            
            if len(training_data) < 50:  # Need minimum data for training
                logger.info("Insufficient training data for update")
                return
            
            # Convert to training format
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                logger.info("No valid training data")
                return
            
            # Train/update models
            self._update_intervention_model(X, y)
            self._update_cost_model(X, y)
            self._update_effectiveness_model(X, y)
            
            # Evaluate performance
            performance = self._evaluate_models()
            self.performance_history.append(performance)
            
            # Save updated models
            self._save_models()
            
            logger.info(f"Incremental update completed. New accuracy: {performance.accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
    
    def _prepare_training_data(self, examples: List[LearningExample]) -> Tuple[List[Dict], List[Any]]:
        """Prepare training data from examples"""
        X = []
        y = []
        
        for example in examples:
            if example.ground_truth is not None:
                X.append(example.input_features)
                y.append(example.ground_truth)
        
        return X, y
    
    def _update_intervention_model(self, X: List[Dict], y: List[Any]):
        """Update intervention recommendation model"""
        try:
            # Convert features to numerical format
            X_numeric = self._features_to_numeric(X)
            
            if len(X_numeric) == 0:
                return
            
            # Train Random Forest for intervention classification
            if 'intervention_model' not in self.models:
                self.models['intervention_model'] = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            
            # Fit model
            self.models['intervention_model'].fit(X_numeric, y)
            
            logger.info("Intervention model updated")
            
        except Exception as e:
            logger.error(f"Error updating intervention model: {e}")
    
    def _update_cost_model(self, X: List[Dict], y: List[Any]):
        """Update cost estimation model"""
        try:
            # Extract cost-related features and targets
            cost_features = []
            cost_targets = []
            
            for i, features in enumerate(X):
                if 'cost' in str(y[i]).lower():
                    cost_features.append(features)
                    cost_targets.append(y[i])
            
            if len(cost_features) < 10:
                return
            
            X_numeric = self._features_to_numeric(cost_features)
            
            if 'cost_model' not in self.models:
                self.models['cost_model'] = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42
                )
            
            self.models['cost_model'].fit(X_numeric, cost_targets)
            
            logger.info("Cost model updated")
            
        except Exception as e:
            logger.error(f"Error updating cost model: {e}")
    
    def _update_effectiveness_model(self, X: List[Dict], y: List[Any]):
        """Update effectiveness prediction model"""
        try:
            # Extract effectiveness-related features
            effectiveness_features = []
            effectiveness_targets = []
            
            for i, features in enumerate(X):
                if 'effectiveness' in str(y[i]).lower() or 'accident' in str(y[i]).lower():
                    effectiveness_features.append(features)
                    effectiveness_targets.append(y[i])
            
            if len(effectiveness_features) < 10:
                return
            
            X_numeric = self._features_to_numeric(effectiveness_features)
            
            if 'effectiveness_model' not in self.models:
                self.models['effectiveness_model'] = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
            
            self.models['effectiveness_model'].fit(X_numeric, effectiveness_targets)
            
            logger.info("Effectiveness model updated")
            
        except Exception as e:
            logger.error(f"Error updating effectiveness model: {e}")
    
    def _features_to_numeric(self, features_list: List[Dict]) -> np.ndarray:
        """Convert feature dictionaries to numerical arrays"""
        try:
            # Simple feature encoding
            numeric_features = []
            
            for features in features_list:
                numeric_row = []
                
                # Categorical features (one-hot encoding)
                categorical_features = ['city', 'state', 'road_type', 'query_type']
                for feature in categorical_features:
                    if feature in features:
                        # Simple hash encoding
                        numeric_row.append(hash(str(features[feature])) % 1000)
                    else:
                        numeric_row.append(0)
                
                # Numerical features
                numerical_features = ['text_length', 'intervention_count']
                for feature in numerical_features:
                    if feature in features:
                        numeric_row.append(float(features[feature]))
                    else:
                        numeric_row.append(0.0)
                
                # Boolean features
                boolean_features = ['has_image', 'has_location']
                for feature in boolean_features:
                    if feature in features:
                        numeric_row.append(1.0 if features[feature] else 0.0)
                    else:
                        numeric_row.append(0.0)
                
                numeric_features.append(numeric_row)
            
            return np.array(numeric_features)
            
        except Exception as e:
            logger.error(f"Error converting features to numeric: {e}")
            return np.array([])
    
    def _evaluate_models(self) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            # Get recent examples for evaluation
            recent_examples = list(self.training_examples)[-1000:]  # Last 1000 examples
            
            if len(recent_examples) < 50:
                return ModelPerformance(
                    model_name="combined",
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    confidence_threshold=self.confidence_threshold,
                    evaluation_timestamp=datetime.now().isoformat(),
                    test_set_size=0
                )
            
            # Prepare test data
            test_examples = [ex for ex in recent_examples if ex.ground_truth is not None]
            
            if len(test_examples) < 10:
                return ModelPerformance(
                    model_name="combined",
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    confidence_threshold=self.confidence_threshold,
                    evaluation_timestamp=datetime.now().isoformat(),
                    test_set_size=len(test_examples)
                )
            
            X_test, y_test = self._prepare_training_data(test_examples)
            X_numeric = self._features_to_numeric(X_test)
            
            if len(X_numeric) == 0:
                return ModelPerformance(
                    model_name="combined",
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    confidence_threshold=self.confidence_threshold,
                    evaluation_timestamp=datetime.now().isoformat(),
                    test_set_size=len(test_examples)
                )
            
            # Evaluate intervention model
            if 'intervention_model' in self.models:
                y_pred = self.models['intervention_model'].predict(X_numeric)
                
                # Convert to binary classification for metrics
                y_test_binary = [1 if 'accepted' in str(label).lower() else 0 for label in y_test]
                y_pred_binary = [1 if 'accepted' in str(pred).lower() else 0 for label in y_pred]
                
                accuracy = accuracy_score(y_test_binary, y_pred_binary)
                precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
                recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
                f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
            else:
                accuracy = precision = recall = f1 = 0.0
            
            return ModelPerformance(
                model_name="intervention_model",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confidence_threshold=self.confidence_threshold,
                evaluation_timestamp=datetime.now().isoformat(),
                test_set_size=len(test_examples)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return ModelPerformance(
                model_name="error",
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                confidence_threshold=self.confidence_threshold,
                evaluation_timestamp=datetime.now().isoformat(),
                test_set_size=0
            )
    
    def _save_feedback(self, feedback: UserFeedback):
        """Save feedback to disk"""
        try:
            feedback_file = self.feedback_path / f"feedback_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Load existing feedback
            existing_feedback = []
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    existing_feedback = json.load(f)
            
            # Add new feedback
            existing_feedback.append(asdict(feedback))
            
            # Save updated feedback
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                model_file = self.models_path / f"{model_name}.joblib"
                joblib.dump(model, model_file)
            
            # Save performance history
            performance_file = self.models_path / "performance_history.json"
            performance_data = [asdict(perf) for perf in self.performance_history]
            
            with open(performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_existing_data(self):
        """Load existing models and data"""
        try:
            # Load models
            for model_file in self.models_path.glob("*.joblib"):
                model_name = model_file.stem
                if model_name != "performance_history":
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded model: {model_name}")
            
            # Load performance history
            performance_file = self.models_path / "performance_history.json"
            if performance_file.exists():
                with open(performance_file, 'r', encoding='utf-8') as f:
                    performance_data = json.load(f)
                
                self.performance_history = [
                    ModelPerformance(**perf) for perf in performance_data
                ]
                logger.info(f"Loaded {len(self.performance_history)} performance records")
            
            # Load recent feedback
            for feedback_file in self.feedback_path.glob("feedback_*.json"):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                
                for feedback_dict in feedback_data:
                    feedback = UserFeedback(**feedback_dict)
                    self.feedback_buffer.append(feedback)
                
                logger.info(f"Loaded {len(feedback_data)} feedback records from {feedback_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        return {
            'total_feedback': len(self.feedback_buffer),
            'total_examples': len(self.training_examples),
            'models_trained': len(self.models),
            'performance_history': len(self.performance_history),
            'latest_accuracy': self.performance_history[-1].accuracy if self.performance_history else 0.0,
            'learning_threshold': self.learning_threshold,
            'confidence_threshold': self.confidence_threshold,
            'active_learning_threshold': self.active_learning_threshold
        }

# Global instance
continuous_learner = None

def get_continuous_learner() -> ContinuousLearner:
    """Get global continuous learner instance"""
    global continuous_learner
    if continuous_learner is None:
        continuous_learner = ContinuousLearner()
    return continuous_learner

def add_user_feedback(feedback: UserFeedback):
    """Convenience function for adding feedback"""
    learner = get_continuous_learner()
    learner.add_user_feedback(feedback)
