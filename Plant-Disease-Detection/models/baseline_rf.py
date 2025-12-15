"""
Random Forest Baseline Model

This module implements a traditional machine learning baseline using:
- Feature extraction from images
- PCA for dimensionality reduction
- Random Forest classifier
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time
import joblib
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestBaseline:
    """
    Random Forest baseline model for plant disease classification.
    
    Pipeline:
    1. Load and resize images
    2. Extract features (flatten pixel values or use color histograms)
    3. Apply PCA for dimensionality reduction
    4. Scale features
    5. Train Random Forest classifier
    
    This serves as a traditional ML baseline to compare against deep learning models.
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: Optional[int] = None,
                 n_components: int = 100,
                 img_size: Tuple[int, int] = (64, 64),
                 feature_type: str = 'pixel',
                 random_state: int = 42):
        """
        Initialize Random Forest baseline.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int, optional): Maximum depth of trees
            n_components (int): Number of PCA components
            img_size (tuple): Image size for feature extraction
            feature_type (str): 'pixel' or 'histogram'
            random_state (int): Random seed
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for RandomForestBaseline")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_components = n_components
        self.img_size = img_size
        self.feature_type = feature_type
        self.random_state = random_state
        
        # Initialize components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.is_fitted = False
        self.training_time = 0
        self.feature_importance = None
        
        logger.info(f"RandomForestBaseline initialized: n_estimators={n_estimators}, "
                   f"n_components={n_components}, feature_type={feature_type}")
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images (np.ndarray): Array of images (N, H, W, C) or list of image paths
            
        Returns:
            np.ndarray: Feature matrix (N, feature_dim)
        """
        if isinstance(images, list):
            # Load images from paths
            images = self._load_images(images)
        
        if self.feature_type == 'pixel':
            features = self._extract_pixel_features(images)
        elif self.feature_type == 'histogram':
            features = self._extract_histogram_features(images)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        logger.info(f"Extracted features: shape={features.shape}")
        return features
    
    def _load_images(self, image_paths: list) -> np.ndarray:
        """Load and resize images from paths."""
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL required for loading images")
        
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize(self.img_size)
                images.append(np.array(img))
            except Exception as e:
                logger.warning(f"Error loading image {path}: {e}")
                # Add blank image
                images.append(np.zeros((*self.img_size, 3)))
        
        return np.array(images)
    
    def _extract_pixel_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract pixel-based features by flattening images.
        
        Args:
            images (np.ndarray): Array of images (N, H, W, C)
            
        Returns:
            np.ndarray: Flattened features (N, H*W*C)
        """
        # Flatten each image
        features = images.reshape(images.shape[0], -1)
        
        # Normalize to [0, 1]
        features = features.astype(np.float32) / 255.0
        
        return features
    
    def _extract_histogram_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            images (np.ndarray): Array of images (N, H, W, C)
            
        Returns:
            np.ndarray: Histogram features (N, bins*3)
        """
        bins = 32  # Number of bins per channel
        features = []
        
        for img in images:
            # Calculate histogram for each channel
            hist_r = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))[0]
            hist_g = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))[0]
            hist_b = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))[0]
            
            # Concatenate histograms
            hist = np.concatenate([hist_r, hist_g, hist_b])
            
            # Normalize
            hist = hist.astype(np.float32) / hist.sum()
            
            features.append(hist)
        
        return np.array(features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X_train (np.ndarray): Training images or features
            y_train (np.ndarray): Training labels
            
        Returns:
            dict: Training results and statistics
        """
        logger.info("Starting Random Forest training...")
        start_time = time.time()
        
        # Extract features if input is images
        if len(X_train.shape) > 2:
            X_train = self.extract_features(X_train)
        
        # Apply PCA
        logger.info("Applying PCA dimensionality reduction...")
        X_train_pca = self.pca.fit_transform(X_train)
        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_variance:.2%}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_pca)
        
        # Train Random Forest
        logger.info("Training Random Forest classifier...")
        self.rf.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        # Get feature importance
        self.feature_importance = self.rf.feature_importances_
        
        # Training accuracy
        y_pred_train = self.rf.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        logger.info(f"Training completed in {self.training_time:.2f}s. "
                   f"Train accuracy: {train_accuracy:.4f}")
        
        return {
            'training_time': self.training_time,
            'train_accuracy': train_accuracy,
            'n_components': self.n_components,
            'explained_variance': explained_variance,
            'feature_importance_available': True
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Images or features to predict
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract features if input is images
        if len(X.shape) > 2:
            X = self.extract_features(X)
        
        # Transform features
        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        
        # Predict
        predictions = self.rf.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Images or features to predict
            
        Returns:
            np.ndarray: Predicted probabilities (N, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract features if input is images
        if len(X.shape) > 2:
            X = self.extract_features(X)
        
        # Transform features
        X_pca = self.pca.transform(X)
        X_scaled = self.scaler.transform(X_pca)
        
        # Predict probabilities
        probabilities = self.rf.predict_proba(X_scaled)
        return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test (np.ndarray): Test images or features
            y_test (np.ndarray): True test labels
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating Random Forest model...")
        
        # Predict
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': self.training_time
        }
        
        logger.info(f"Test Results - Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_k: int = 20) -> Dict:
        """
        Get top feature importances.
        
        Args:
            top_k (int): Number of top features to return
            
        Returns:
            dict: Feature importance information
        """
        if self.feature_importance is None:
            return {'error': 'Model not trained or feature importance not available'}
        
        # Get top k indices
        top_indices = np.argsort(self.feature_importance)[-top_k:][::-1]
        top_importance = self.feature_importance[top_indices]
        
        return {
            'top_indices': top_indices.tolist(),
            'top_importance': top_importance.tolist(),
            'all_importance': self.feature_importance.tolist()
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Nothing to save.")
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'rf': self.rf,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'n_components': self.n_components,
                'img_size': self.img_size,
                'feature_type': self.feature_type,
                'random_state': self.random_state
            },
            'training_time': self.training_time,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestBaseline':
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            RandomForestBaseline: Loaded model
        """
        model_data = joblib.load(filepath)
        
        # Create instance
        config = model_data['config']
        model = cls(**config)
        
        # Restore components
        model.pca = model_data['pca']
        model.scaler = model_data['scaler']
        model.rf = model_data['rf']
        model.training_time = model_data['training_time']
        model.feature_importance = model_data['feature_importance']
        model.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_components': self.n_components,
            'img_size': self.img_size,
            'feature_type': self.feature_type,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
