# -*- coding: utf-8 -*-
"""
A collection of metrics for fairness assessment in machine learning.
"""

# Author: Gabriel Rozzonelli

import numpy as np


def dfpr(y_true, y_pred, sensitive=None):
    """DFPR classification score.
    
    This metrics returns the difference between sensitive and non-sensitive
    False Positive Rates, which is useful to quantify disparate mistreatment.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive_samples : array-like of shape (n_sensitive_samples,), default=None
        Indices of the sensitive samples. The array should be filled
        with integers.
        
    Returns
    -------
    score : float
        DFPR score. The closer to 0, the lesser is disparate mistreatment.
    """
    # Prepare masks
    s = sensitive if sensitive is not None else np.zeros(len(y_pred)).astype(bool)
    wrong = y_true != y_pred
    positive = y_true == 1
    
    sum_pos_s = np.sum(positive[s])
    sum_pos_ns = np.sum(positive[~s])
    a = np.sum(wrong[s] & positive[s]) / sum_pos_s if sum_pos_s > 0 else 1
    b = np.sum(wrong[~s] & positive[~s]) / sum_pos_ns if sum_pos_ns > 0 else 1
    score = a - b
    
    return score
    
def dfnr(y_true, y_pred, sensitive=None):
    """DFNR classification score.
    
    This metrics returns the difference between sensitive and non-sensitive
    False Negative Rates, which is useful to quantify disparate mistreatment.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive_samples : array-like of shape (n_sensitive_samples,), default=None
        Indices of the sensitive samples. The array should be filled
        with integers.
        
    Returns
    -------
    score : float
        DFNR score. The closer to 0, the lesser is disparate mistreatment.
    """
    # Prepare masks
    s = sensitive if sensitive is not None else np.zeros(len(y_pred)).astype(bool)
    wrong = y_true != y_pred
    negative = y_true == 0

    sum_neg_s = np.sum(negative[s])
    sum_neg_ns = np.sum(negative[~s])
    a = np.sum(wrong[s] & negative[s]) / sum_neg_s if sum_neg_s > 0 else 1
    b = np.sum(wrong[~s] & negative[~s]) / sum_neg_ns if sum_neg_ns > 0 else 1
    score = a - b
    
    return score

def p_rule(y_true, y_pred, sensitive):
    """p% rule classification score.
    
    This metrics is an empirical rule, which is useful to quantify
    disparate impact.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive_samples : array-like of shape (n_sensitive_samples,), default=None
        Indices of the sensitive samples. The array should be filled
        with intergers.
        
    Returns
    -------
    score : float
        p% rule score. The closer to 1, the lesser is disparate impact.
    """
    # Prepare masks
    s = sensitive if sensitive is not None else np.zeros(len(y_pred)).astype(bool)
    positive_pred = y_pred == 1
    
    a = np.sum(positive_pred[s]) / np.sum(positive_pred[~s]) \
        * np.sum(~s) / np.sum(s)
        
    if a == 0:
        score = 0
    else:
        score = min(a, 1/a)
        
    return score
