"""
This module contains several preprocessed datasets for fairness
classification assessment.
"""

from .adult import AdultDataset
from .bank import BankDataset
from .compas import COMPASDataset
from .kdd import KDDDataset

__all__ = [
    "AdultDataset",
    "BankDataset",
    "COMPASDataset",
    "KDDDataset",
]
