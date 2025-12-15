"""
Models Module for Plant Disease Detection

This module contains baseline and advanced models:
- Random Forest (traditional ML)
- Simple CNN (baseline deep learning)
- Transfer Learning (ResNet, VGG, etc.)
- Existing CNN (from original repository)
"""

from .baseline_rf import RandomForestBaseline
from .simple_cnn import SimpleCNN
from .transfer_learning import TransferLearningModel

__all__ = [
    'RandomForestBaseline',
    'SimpleCNN',
    'TransferLearningModel'
]
