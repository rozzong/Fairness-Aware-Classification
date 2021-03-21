# -*- coding: utf-8 -*-
"""
The Adult dataset.
"""

from sklearn.metrics import accuracy_score

from .dataset import Dataset
from ..metrics import p_rule_score


class AdultDataset(Dataset):
    
    RAW_FILENAME = "adult.csv"
    CLEAN_FILENAME = "adult_preprocessed.csv"
    PARSE_KWARGS = {"sep": ", ", "engine": "python"}
    NA_VALUES = ["?"]
    FEATURES_TO_DROP = None
    FEATURES_TO_KEEP = None
    FORCE_NUM = None
    FORCE_CAT = None
    TARGET_LABEL = "income"
    SENSITIVE_FEATURES = ["sex"]
    SENSITIVE_VALUES = [1]
    
    def __init__(self):
        super().__init__()
        
    def _custom(self):
        pass
        
    def _prepare(self):
        self._scale()
        self._encode_one_hot()
    
    def objective(self, y_true, y_pred, sensitive):
        acc = accuracy_score(y_true, y_pred)
        p = p_rule_score(y_true, y_pred, sensitive)
       
        return acc + p
