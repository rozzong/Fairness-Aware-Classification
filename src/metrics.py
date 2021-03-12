# -*- coding: utf-8 -*-

import numpy as np


def dfpr(y_true, y_pred, sensitive_samples=None):
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
        with intergers.
        
    Returns
    -------
    score : float
        DFPR score. The closer to 0, the lesser is disparate mistreatment.
    """
    # Prepare masks
    s = sensitive_samples if sensitive_samples != None else np.arange(0)
    wrong = y_true != y_pred
    positive = y_true == 1
    
    a = np.sum(wrong[s] & positive[s]) / np.sum(positive[s])
    b = np.sum(wrong[~s] & positive[~s]) / np.sum(positive[~s])
    score = a - b
    
    return score
    
def dfnr(y_true, y_pred, sensitive_samples=None):
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
        with intergers.
        
    Returns
    -------
    score : float
        DFNR score. The closer to 0, the lesser is disparate mistreatment.
    """
    # Prepare masks
    s = sensitive_samples if sensitive_samples != None else np.arange(0)
    wrong = y_true != y_pred
    negative = y_true == 0
    
    a = np.sum(wrong[s] & negative[s]) / np.sum(negative[s])
    b = np.sum(wrong[~s] & negative[~s]) / np.sum(negative[~s])
    score = a - b
    
    return score
    
def p_rule(y_true, y_pred, sensitive_samples):
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
    s = sensitive_samples if sensitive_samples != None else np.arange(0)
    positive_pred = y_pred == 1
    
    a = np.sum(positive_pred[s]) / np.sum(positive_pred[~s]) \
        * np.sum(~s) / np.sum(s)
    score = min(a, 1/a)
        
    return score
