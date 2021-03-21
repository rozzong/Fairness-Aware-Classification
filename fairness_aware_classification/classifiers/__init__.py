"""
This module contains several fairness aware classifiers.
"""

from .adafair import AdaFairClassifier
from .adaptive_weights import AdaptiveWeightsClassifier
from .smoteboost import SMOTEBoostClassifier

__all__ = [
    "AdaFairCLassifier",
    "AdaptiveWeightsClassifier",
    "SMOTEBoostClassifier",
]
