"""
This module contains several fairness aware classifiers.
"""

from .adafair import AdaFairClassifier
from .adaptive_weights import AdaptiveWeightsClassifier

__all__ = [
    "AdaFairCLassifier",
    "AdaptiveWeightsClassifier",
]
