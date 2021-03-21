# -*- coding: utf-8 -*-
"""
The COMPAS dataset.
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from ..metrics import dfpr_score, dfnr_score


class COMPASDataset(Dataset):
    
    RAW_FILENAME = "compas-scores-two-years.csv"
    CLEAN_FILENAME = "compas-scores-two-years_preprocessed.csv"
    PARSE_KWARGS = {"index_col": 0}
    NA_VALUES = None
    FEATURES_TO_DROP = None
    FEATURES_TO_KEEP = [
        "sex",
        "age_cat",
        "race",
        "priors_count",
        "c_charge_degree",
    ]
    FORCE_NUM = None
    FORCE_CAT = ["two_year_recid"]
    TARGET_LABEL = "two_year_recid"
    SENSITIVE_FEATURES = ["sex"]
    SENSITIVE_VALUES = [0]
    
    def __init__(self):
        super().__init__()
        
    def _custom(self):
        # Keep only specific value groups
        values = ["African-American", "Caucasian"]
        self._df = self._df.loc[self._df["race"].isin(values)]
        
    def _prepare(self):       
        self._scale()
        self._encode_one_hot()
    
    def objective(self, y_true, y_pred, sensitive):
        acc = accuracy_score(y_true, y_pred)
        dfpr = dfpr_score(y_true, y_pred, sensitive)
        dfnr = dfnr_score(y_true, y_pred, sensitive)
        
        return 2 * acc - abs(dfpr) - abs(dfnr)
