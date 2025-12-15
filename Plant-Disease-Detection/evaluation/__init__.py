"""
Evaluation Module for Plant Disease Detection

This module provides comprehensive model evaluation tools including:
- Metrics calculation (accuracy, precision, recall, F1)
- Confusion matrix generation
- ROC curves and AUC
- Model comparison
"""

from .metrics import ModelEvaluator
from .visualizations import ResultsVisualizer
from .compare_models import ModelComparison

__all__ = ['ModelEvaluator', 'ResultsVisualizer', 'ModelComparison']
