"""
Intelligent Agent for Plant Disease Detection

This module implements a goal-based learning agent that perceives plant leaf images,
makes decisions based on confidence thresholds, and recommends treatments.

Agent Type: Goal-Based Learning Agent
Components:
- Perception: Image input processing
- State: Internal knowledge representation
- Goal: Accurate disease identification (>90% confidence)
- Actions: Classify, request more info, recommend treatment
- Performance Measure: Classification accuracy and confidence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantDiseaseAgent:
    """
    Goal-based learning agent for plant disease detection.
    
    The agent follows a perception-action cycle:
    1. Perceive: Receive plant leaf image
    2. Think: Process image and generate predictions
    3. Decide: Choose action based on confidence and goals
    4. Act: Execute the chosen action
    
    Attributes:
        model: Disease classification model
        confidence_threshold (float): Minimum confidence for classification
        state (dict): Internal state representation
        goal (str): Agent's primary objective
        actions (list): Available actions
        treatment_searcher: Search algorithm for treatment recommendation
    """
    
    def __init__(self, model=None, confidence_threshold: float = 0.90, 
                 treatment_searcher=None):
        """
        Initialize the Plant Disease Agent.
        
        Args:
            model: Pre-trained disease classification model
            confidence_threshold (float): Confidence threshold for classification (0-1)
            treatment_searcher: Treatment search algorithm (e.g., A* searcher)
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.treatment_searcher = treatment_searcher
        
        # Agent state
        self.state = {
            'current_image': None,
            'preprocessed_image': None,
            'predictions': None,
            'confidence': 0.0,
            'predicted_disease': None,
            'action_taken': None,
            'treatment_plan': None,
            'history': []
        }
        
        # Agent goal
        self.goal = "accurate_disease_identification"
        
        # Available actions
        self.actions = [
            'CLASSIFY_AND_TREAT',
            'REQUEST_MORE_IMAGES',
            'REQUEST_EXPERT_REVIEW',
            'SEARCH_TREATMENT',
            'NO_DISEASE_DETECTED'
        ]
        
        logger.info(f"PlantDiseaseAgent initialized with confidence threshold: {confidence_threshold}")
    
    def perceive(self, image: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Agent receives plant image as input (perception phase).
        
        Args:
            image (np.ndarray): Plant leaf image
            metadata (dict, optional): Additional information (location, date, etc.)
        """
        self.state['current_image'] = image
        self.state['metadata'] = metadata or {}
        
        logger.info(f"Agent perceived new image of shape: {image.shape}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for model input.
        
        Args:
            image (np.ndarray): Raw image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # This will be implemented based on the model's requirements
        # For now, placeholder implementation
        if image is None:
            return None
        
        preprocessed = image.copy()
        # Add normalization, resizing, etc. based on model requirements
        
        self.state['preprocessed_image'] = preprocessed
        return preprocessed
    
    def think(self) -> Dict[str, Any]:
        """
        Agent processes image and makes predictions (thinking phase).
        
        Returns:
            dict: Predictions with confidence scores
        """
        if self.model is None:
            logger.warning("No model loaded. Using mock predictions.")
            return self._mock_predictions()
        
        # Preprocess image
        preprocessed_image = self.preprocess_image(self.state['current_image'])
        
        # Make predictions
        predictions = self.model.predict(preprocessed_image)
        
        # Store predictions in state
        self.state['predictions'] = predictions
        self.state['confidence'] = float(np.max(predictions))
        self.state['predicted_disease'] = self._get_disease_name(predictions)
        
        logger.info(f"Predictions generated: {self.state['predicted_disease']} "
                   f"with confidence {self.state['confidence']:.2f}")
        
        return {
            'disease': self.state['predicted_disease'],
            'confidence': self.state['confidence'],
            'all_predictions': predictions
        }
    
    def decide(self) -> str:
        """
        Goal-based decision making (decision phase).
        
        Decision logic:
        - If confidence >= threshold: CLASSIFY_AND_TREAT
        - If confidence < threshold but > 0.5: REQUEST_MORE_IMAGES
        - If confidence <= 0.5: REQUEST_EXPERT_REVIEW
        - If disease is 'healthy': NO_DISEASE_DETECTED
        
        Returns:
            str: Selected action
        """
        confidence = self.state['confidence']
        disease = self.state['predicted_disease']
        
        # Check if goal is satisfied
        if self._is_goal_satisfied():
            if disease and 'healthy' in disease.lower():
                action = 'NO_DISEASE_DETECTED'
            else:
                action = 'CLASSIFY_AND_TREAT'
        elif confidence >= 0.5:
            action = 'REQUEST_MORE_IMAGES'
        else:
            action = 'REQUEST_EXPERT_REVIEW'
        
        self.state['action_taken'] = action
        logger.info(f"Agent decided action: {action}")
        
        return action
    
    def act(self, action: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the decided action (action phase).
        
        Args:
            action (str, optional): Action to execute. If None, uses decided action.
            
        Returns:
            dict: Result of the action
        """
        if action is None:
            action = self.state['action_taken']
        
        result = {}
        
        if action == 'CLASSIFY_AND_TREAT':
            result = self._classify_and_treat()
        elif action == 'REQUEST_MORE_IMAGES':
            result = self._request_more_images()
        elif action == 'REQUEST_EXPERT_REVIEW':
            result = self._request_expert_review()
        elif action == 'SEARCH_TREATMENT':
            result = self._search_treatment()
        elif action == 'NO_DISEASE_DETECTED':
            result = self._no_disease_detected()
        else:
            result = {'status': 'error', 'message': f'Unknown action: {action}'}
        
        # Update history
        self.state['history'].append({
            'action': action,
            'result': result,
            'confidence': self.state['confidence'],
            'disease': self.state['predicted_disease']
        })
        
        return result
    
    def run(self, image: np.ndarray, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete perception-action cycle.
        
        Args:
            image (np.ndarray): Plant leaf image
            metadata (dict, optional): Additional information
            
        Returns:
            dict: Final result with disease, treatment, and recommendations
        """
        # 1. Perceive
        self.perceive(image, metadata)
        
        # 2. Think
        predictions = self.think()
        
        # 3. Decide
        action = self.decide()
        
        # 4. Act
        result = self.act(action)
        
        return result
    
    def _is_goal_satisfied(self) -> bool:
        """
        Check if the agent's goal is satisfied.
        
        Returns:
            bool: True if confidence meets or exceeds threshold
        """
        return self.state['confidence'] >= self.confidence_threshold
    
    def _get_disease_name(self, predictions: np.ndarray) -> str:
        """
        Get disease name from prediction array.
        
        Args:
            predictions (np.ndarray): Prediction probabilities
            
        Returns:
            str: Disease name
        """
        # This should map to actual disease classes
        # Placeholder implementation
        if predictions is None:
            return "Unknown"
        
        predicted_idx = np.argmax(predictions)
        return f"Disease_Class_{predicted_idx}"
    
    def _classify_and_treat(self) -> Dict[str, Any]:
        """
        Classify disease and recommend treatment.
        
        Returns:
            dict: Classification results and treatment plan
        """
        disease = self.state['predicted_disease']
        confidence = self.state['confidence']
        
        # Search for treatment if searcher is available
        treatment_plan = None
        if self.treatment_searcher:
            treatment_plan = self.treatment_searcher.search(disease)
            self.state['treatment_plan'] = treatment_plan
        
        return {
            'status': 'success',
            'action': 'CLASSIFY_AND_TREAT',
            'disease': disease,
            'confidence': confidence,
            'treatment_plan': treatment_plan,
            'message': f"Detected {disease} with {confidence:.2%} confidence"
        }
    
    def _request_more_images(self) -> Dict[str, Any]:
        """
        Request additional images for better classification.
        
        Returns:
            dict: Request details
        """
        return {
            'status': 'needs_more_data',
            'action': 'REQUEST_MORE_IMAGES',
            'confidence': self.state['confidence'],
            'disease': self.state['predicted_disease'],
            'message': f"Confidence ({self.state['confidence']:.2%}) below threshold. "
                      f"Please provide additional images of the affected area."
        }
    
    def _request_expert_review(self) -> Dict[str, Any]:
        """
        Request expert review for low-confidence predictions.
        
        Returns:
            dict: Expert review request
        """
        return {
            'status': 'expert_review_needed',
            'action': 'REQUEST_EXPERT_REVIEW',
            'confidence': self.state['confidence'],
            'disease': self.state['predicted_disease'],
            'message': f"Low confidence ({self.state['confidence']:.2%}). "
                      f"Expert review recommended."
        }
    
    def _search_treatment(self) -> Dict[str, Any]:
        """
        Search for optimal treatment using search algorithm.
        
        Returns:
            dict: Treatment search results
        """
        if not self.treatment_searcher:
            return {
                'status': 'error',
                'message': 'No treatment searcher available'
            }
        
        disease = self.state['predicted_disease']
        treatment_plan = self.treatment_searcher.search(disease)
        
        return {
            'status': 'success',
            'action': 'SEARCH_TREATMENT',
            'disease': disease,
            'treatment_plan': treatment_plan
        }
    
    def _no_disease_detected(self) -> Dict[str, Any]:
        """
        Handle healthy plant detection.
        
        Returns:
            dict: Healthy plant confirmation
        """
        return {
            'status': 'success',
            'action': 'NO_DISEASE_DETECTED',
            'disease': self.state['predicted_disease'],
            'confidence': self.state['confidence'],
            'message': "Plant appears healthy. No treatment needed."
        }
    
    def _mock_predictions(self) -> Dict[str, Any]:
        """
        Generate mock predictions for testing.
        
        Returns:
            dict: Mock prediction results
        """
        # Mock predictions for testing
        mock_probs = np.random.dirichlet(np.ones(34))  # 34 classes in Pakistan dataset (Rice, Cotton, Wheat, Mango + PlantVillage)
        self.state['predictions'] = mock_probs
        self.state['confidence'] = float(np.max(mock_probs))
        self.state['predicted_disease'] = f"Disease_Class_{np.argmax(mock_probs)}"
        
        return {
            'disease': self.state['predicted_disease'],
            'confidence': self.state['confidence'],
            'all_predictions': mock_probs
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.
        
        Returns:
            dict: Current state
        """
        return self.state.copy()
    
    def reset(self) -> None:
        """Reset agent state."""
        self.state = {
            'current_image': None,
            'preprocessed_image': None,
            'predictions': None,
            'confidence': 0.0,
            'predicted_disease': None,
            'action_taken': None,
            'treatment_plan': None,
            'history': []
        }
        logger.info("Agent state reset")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate agent performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        history = self.state['history']
        
        if not history:
            return {'message': 'No history available'}
        
        total_actions = len(history)
        successful_classifications = sum(
            1 for h in history 
            if h['action'] == 'CLASSIFY_AND_TREAT'
        )
        avg_confidence = np.mean([h['confidence'] for h in history])
        
        return {
            'total_actions': total_actions,
            'successful_classifications': successful_classifications,
            'classification_rate': successful_classifications / total_actions if total_actions > 0 else 0,
            'average_confidence': avg_confidence,
            'goal_satisfaction_rate': successful_classifications / total_actions if total_actions > 0 else 0
        }
