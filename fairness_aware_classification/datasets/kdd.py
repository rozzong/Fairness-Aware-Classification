# -*- coding: utf-8 -*-
"""
The KDD census income dataset.
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from ..metrics import p_rule_score


class KDDDataset(Dataset):
    
    RAW_FILENAME = "kdd_census_income.csv"
    CLEAN_FILENAME = "kdd_census_income_preprocessed_ordinal.csv"
    PARSE_KWARGS = {"sep": ", ", "engine": "python"}
    NA_VALUES = ["?"]
    FEATURES_TO_DROP = None
    FEATURES_TO_KEEP = None
    FORCE_NUM = None
    FORCE_CAT = [
        "industry code",
        "occupation code",
        "own business or self employed",
    ]
    TARGET_LABEL = "income"
    SENSITIVE_FEATURES = ["sex"]
    SENSITIVE_VALUES = [0]
    
    def __init__(self):
        super().__init__()
        
    def _custom(self):
        pass
        
    def _prepare(self):       
        self._scale()
        self._encode_ordinal()
    
    def objective(self, y_true, y_pred, sensitive):
        acc = accuracy_score(y_true, y_pred)
        p = p_rule_score(y_true, y_pred, sensitive)
       
        return acc + p
