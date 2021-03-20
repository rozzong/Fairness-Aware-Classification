# -*- coding: utf-8 -*-
"""
A collection of metrics for fairness assessment in machine learning.
"""

# Authors: Gabriel Rozzonelli, Bekarys Nurtay

import numpy as np

# AdaptiveWeights metrics
def dfpr_score(y_true, y_pred, sensitive=None):
    """DFPR classification score.
    
    This metrics returns the difference between sensitive and non-sensitive
    False Positive Rates, which is useful to quantify disparate mistreatment.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive. If set to None, no sample is considered as 
        sensitive.
        
    Returns
    -------
    score : float
        DFPR score. The closer to 0, the lesser is disparate mistreatment.
    """
    if sensitive is None:
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive
    
    wrong = y_true != y_pred
    negative = y_true == 0
    sum_neg_s = np.sum(negative[s])
    sum_neg_ns = np.sum(negative[~s])
    
    a = np.sum(wrong[s] & negative[s]) / sum_neg_s if sum_neg_s > 0 else 1
    b = np.sum(wrong[~s] & negative[~s]) / sum_neg_ns if sum_neg_ns > 0 else 1
    
    score = a - b
    
    return score
    
def dfnr_score(y_true, y_pred, sensitive=None):
    """DFNR classification score.
    
    This metrics returns the difference between sensitive and non-sensitive
    False Negative Rates, which is useful to quantify disparate mistreatment.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive. If set to None, no sample is considered as 
        sensitive.
        
    Returns
    -------
    score : float
        DFNR score. The closer to 0, the lesser is disparate mistreatment.
    """
    if sensitive is None:
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive
    
    wrong = y_true != y_pred
    positive = y_true == 1
    
    sum_pos_s = np.sum(positive[s])
    sum_pos_ns = np.sum(positive[~s])
    
    a = np.sum(wrong[s] & positive[s]) / sum_pos_s if sum_pos_s > 0 else 1
    b = np.sum(wrong[~s] & positive[~s]) / sum_pos_ns if sum_pos_ns > 0 else 1
    
    score = a - b
    
    return score
    
def eq_odds_score(y_true, y_pred, sensitive):
    """Equalized Odds classification score.
    
    This metrics returns the difference between absolute values of the
    DFPR and the DFNR, which is useful to quantify disparate mistreatment.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive. If set to None, no sample is considered as 
        sensitive.
        
    Returns
    -------
    score : float
        Equalized Odds score. The closer to 0, the lesser is disparate mistreatment.
    """
    dfpr = dfpr_score(y_true, y_pred, sensitive)  
    dfnr = dfnr_score(y_true, y_pred, sensitive)
    
    return  abs(dfpr) + abs(dfnr)

def p_rule_score(y_true, y_pred, sensitive=None):
    """p% rule classification score.
    
    This metrics is an empirical rule, which is useful to quantify
    disparate impact.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive. If set to None, no sample is considered as 
        sensitive.
        
    Returns
    -------
    score : float
        p% rule score. The closer to 1, the lesser is disparate impact.
    """
    if sensitive is None:
        s = np.zeros(len(y_pred)).astype(bool)
    else:
        s = sensitive

    positive_pred = y_pred == 1
    
    a = np.sum(positive_pred[s]) / np.sum(positive_pred[~s]) \
        * np.sum(~s) / np.sum(s)
        
    if a == 0:
        score = 0
    else:
        score = min(a, 1/a)
        
    return score
    
def er_score(y_true, y_pred):
    """Error Rate score.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    Returns
    -------
    score : float
        The Error Rate.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    score = (fn + fp) / (tp + tn + fn + fp)
    
    return score

def ber_score(y_true, y_pred):
    """Balanced Error Rate score.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    Returns
    -------
    score : float
        The Balanced Error Rate.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    score = 1 - (tp / (tp+fn) + tn / (tn+fp)) / 2
    
    return score
    
# TODO: Check that here, sensitive == True means protected (non-sensitive)    
def tpr_protected_score(y, y_pred, sensitive):
    """True Positive Rate score for the protected group.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive.
        
    Returns
    -------
    score : float
        The True Positive Rate for the protected group.
    """
    y = y[sensitive == 0]
    y_pred = y_pred[sensitive == 0]
    
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    
    score = tp / (tp + fn)
    
    return score

def tpr_unprotected_score(y, y_pred, sensitive):
    """True Positive Rate score for the unprotected group.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive.
        
    Returns
    -------
    score : float
        The True Positive Rate for the unprotected group.
    """
    y = y[sensitive == 1]
    y_pred = y_pred[sensitive == 1]
    
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    
    score = tp / (tp + fn)
    
    return score

def tnr_protected_score(y, y_pred, sensitive):
    """True Negative Rate score for the protected group.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive.
        
    Returns
    -------
    score : float
        The True Negative Rate for the protected group.
    """
    y = y[sensitive == 0]
    y_pred = y_pred[sensitive == 0]
    
    tn = ((y == 0) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    
    score = tn / (tn + fp)
    
    return score

def tnr_unprotected_score(y, y_pred, sensitive):
    """True Negative Rate score for the unprotected group.
    
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
        
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
        
    sensitive : array-like of shape (n_samples,), default=None
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive.
        
    Returns
    -------
    score : float
        The True Negative Rate for the unprotected group.
    """
    y = y[sensitive == 1]
    y_pred = y_pred[sensitive == 1]
    
    tn = ((y == 0) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    
    score = tn / (tn + fp)
    
    return score
