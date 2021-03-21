# -*- coding: utf-8 -*-
"""
The Bank marketing dataset.
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from .dataset import Dataset
from ..metrics import p_rule_score


class BankDataset(Dataset):
    
    RAW_FILENAME = "bank_additional_full.csv"
    CLEAN_FILENAME = "bank_additional_full_preprocessed.csv"
    PARSE_KWARGS = {"sep": ";"}
    NA_VALUES = ["unknown"]
    FEATURES_TO_DROP = None
    FEATURES_TO_KEEP = None
    FORCE_NUM = None
    FORCE_CAT = None
    TARGET_LABEL = "y"
    SENSITIVE_FEATURES = ["married"]
    SENSITIVE_VALUES = [1]
    
    def __init__(self):
        super().__init__()
        
    def _custom(self):
        # Add a new column telling whether or not a customer was contacted 
        self._df = self._df.assign(
            contacted=pd.Series(["yes"]*len(self._df)).values
        )
        self._df.loc[self._df["pdays"] == 999, "contacted"] = "no"
        print(self._df["contacted"])
        
    def _prepare(self):       
        self._scale()
        self._encode_one_hot()
    
    def objective(self, y_true, y_pred, sensitive):
        acc = accuracy_score(y_true, y_pred)
        p = p_rule_score(y_true, y_pred, sensitive)
       
        return acc + p
