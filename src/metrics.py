# -*- coding: utf-8 -*-

import numpy as np


def dfpr(y_true, y_pred, sensitive_samples):
    """
    Compute the DFPR.
    """
    # Prepare masks
    sensitive = np.isin(range(len(y_true)), sensitive_samples)
    correctly_classified = y_true == y_pred
    positive = y_true == 1
    
    a = np.sum(correctly_classified[sensitive]==0 & positive[sensitive]==0) / np.sum(positive[sensitive]==0)
    b = np.sum(correctly_classified[~sensitive]==0 & positive[~sensitive]==0) / np.sum(positive[~sensitive]==0)
    
    return a - b
    
def dfnr(y_true, y_pred, sensitive_samples):
    """
    Compute the DFNR.
    """
    # Prepare masks
    sensitive = np.isin(range(len(y_true)), sensitive_samples)
    positive = y_true == 1
    positive_pred = y_pred == 1
    
    a = np.sum(positive_pred[sensitive]==0 & positive[sensitive]==1) / np.sum(positive[sensitive]==1)
    b = np.sum(positive_pred[~sensitive]==0 & positive[~sensitive]==1) / np.sum(positive[~sensitive]==1)
    
    return a - b
    
def p_rule(y_true, y_pred, sensitive_samples):
    """
    Compute the p-rule.
    """
    # Prepare masks
    sensitive = np.isin(range(len(y_true)), sensitive_samples)
    positive_pred = y_pred == 1
    
    a = np.sum(positive_pred[sensitive]) / np.sum(positive_pred[~sensitive]) \
        * np.sum(~sensitive) / np.sum(sensitive)
        
    return min(a, 1/a)
