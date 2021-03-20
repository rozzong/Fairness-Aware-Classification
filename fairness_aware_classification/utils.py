# -*- coding: utf-8 -*-
"""
A collection of useful functions for the project.
"""

# Author: Gabriel Rozzonelli

import numpy as np
import pandas as pd


def sensitive_mask_from_samples(sensitive_samples, n_samples):
    """Sensitive mask from samples.
    
    Parameters
    ----------       
    sensitive_samples : ndarray of shape (n_sensitive_samples,)
        Indices of the sensitive samples.
        
    n_samples : int
        The number of samples in the original set.
        
    Returns
    -------
    sensitive : ndarray of shape (n_samples,)
        The mask, whose values are True where the corresponding sample is
        known to be sensitive.
    """    
    sensitive = np.zeros(n_samples).astype(bool)
    sensitive[sensitive_samples] = True
    
    return sensitive
    
def sensitive_mask_from_features(X, sensitive_features, sensitive_values=None):
    """Sensitive mask from samples.
    
    This function returns a Boolean mask whose length is the number of
    provided samples, based on specified sensitive features.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
        
    sensitive_features : array-like of shape (n_sensitive_features,)
        Indices or labels of the sensitive features.
        
    sensitive_values : array-like of shape (n_sensitive_features,), default=None
        Values raising a sensitive flag for the corresponding features in
        `sensitive_features`. If set to None, the samples are assumed to
        be sensitive if at least one of its features is equal to True.
        
    Returns
    -------
    sensitive : array-like of shape (n_samples,)
        The mask, whose values are True where the corresponding sample is
        known to be sensitive.
    """
    X_ = X.copy()
    
    # Create the mask
    sensitive = np.zeros(len(X_))
    
    # If no values are specified, set the sensitive criteria to True
    if sensitive_values is None:
        sensitive_values = np.ones_like(sensitive_features).astype(bool)
        
    if isinstance(X_, pd.DataFrame):
        for f, v in zip(sensitive_features, sensitive_values):
            sensitive = sensitive | (X_[f] == v)
        
    elif isinstance(X, np.ndarray):
        for f, v in zip(sensitive_features, sensitive_values):
            sensitive = sensitive | (X_[:,f] == v)
            
    else:
        raise ValueError("X should be either a np.ndarray or a " \
                         "pd.DataFrame, but is a {}.".format(type(X)))
    
    return sensitive
    
def spy(func):
    """Decorator to spy on function outputs"""
    def wrapped(*args, **kwargs):
        y = func(*args, **kwargs)
        print(y)
        return y
    return wrapped
