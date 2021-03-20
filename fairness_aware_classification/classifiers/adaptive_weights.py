# -*- coding: utf-8 -*-
"""
The implementation of Adaptive Sensitive Reweighting.
"""

# Author: Gabriel Rozzonelli
# Based on the work of of the following paper:
# [1] E. Krasanakis, E. Spyromitros-Xioufis, S. Papadopoulos, et Y. Kompatsiaris,
#     « Adaptive Sensitive Reweighting to Mitigate Bias in Fairness-aware
#     Classification ».

import warnings 

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize

from ..utils import spy


def exp_loss(e, b):
    """
    Exponential loss
    """
    return np.exp(b*e)

# TODO: Check problems with CallbackCollector not being systematically called
# TODO: Make things more proper
class CallbackCollector:

    def __init__(self, func, threshold=1e-3):
        self._func  = func
        self._threshold = threshold
        self._last_x = None
        self._last_val = None

    def __call__(self, x):
        self._last_x = x
        val = self._func(x)
        print("Stored", x, "for", val)
        if self._last_val is None:
            self._last_val = val
        elif abs(val-self._last_val) <= self._threshold:
            # ~ return True
            raise StopIteration
        else:
            self._last_val = val
        return False
    
class AdaptiveWeightsClassifier(BaseEstimator, ClassifierMixin):
    """Adapative Weights Classifier (AdaptiveWeightsClassifier)
    
    Performs a weighted classification upon a base classifier to preserve
    some fairness properties.
    
    Parameters
    ----------
    base_estimator : object
        The base estimator for which the adaptive weighting is performed.
        Support for sample weighting is required.
    
    criterion : function
        Function criterion to perform the DIRECT optimization with. This
        function should be of the form:
        ``obj = criterion(y_true, y_pred, sensitive_samples)``
        
    loss : str or function, default="exp"
        Convex loss function as ``l = loss(e, beta)``, such that beta is
        its Lipschitz constant. The default loss function is the
        exponential loss: ``l = exp(b*e)``.
        
    eps : float, default=0.1
        Tolerance used as a stopping criterion while computing weights.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator for which the adaptive weighting is performed.
        
    best_params_ : ndarray of shape (4,)
        Best found parameters by the optimization scheme, ordered as
        np.array([a_s, a_ns, b_s, b_ns]).
        
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
        
    n_features_in_: int
        The number of features the classifier was fit for.
    """
    
    _required_parameters = ["base_estimator", "criterion"]
    
    def __init__(self, base_estimator, criterion, loss="exp", eps=0.1):
        self.base_estimator = base_estimator
        self.criterion = criterion            
        self.loss = loss
        self.eps = eps
        
    def fit(self, X, y, sensitive=None):
        """Build a weighted classifier from the training set (X, y) and
        known sensitive samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        sensitive : array-like of shape (n_samples,), default=None
            Mask indicating which samples are sensitive. The array should
            contain boolean values, where True indicates that the corresponding
            sample is sensitive. If set to None, no sample is considered as 
            sensitive.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Perform the arrays
        # TODO: Investigate sparse matrices compatibility
        X, y = check_X_y(X, y, accept_large_sparse=False)
        
        # Check that the base estimator is an classifier
        if not is_classifier(self.base_estimator):
            raise ValueError("The base estimator should be a classifier, " \
                             "but it is not.")
        
        # Check that the it is a binary classification problem
        check_classification_targets(y)
        n_classes = len(np.unique(y))
        if type_of_target(y) != "binary":
            raise ValueError("Target values should belong to a binary set, " \
                             "but {} classes were found.".format(n_classes))
        
        # Handle the case with no specified sensitive sample
        if sensitive is None:
            warnings.warn("Argument `sensitive` is not set. " \
                          "Setting all samples as non-sensitives.")
            sensitive = np.zeros(len(y)).astype(bool)
        
        # Save number of features
        self.n_features_in_ = X.shape[1]
        
        # Save classes
        self.classes_, y = np.unique(y, return_inverse=True)
        
        # Update the convex loss function if needed, or check it
        if self.loss is "exp":
            self._loss = exp_loss
        elif callable(self.loss):
            self._loss = self.loss
        else:
            raise ValueError("Argument 'loss' is not valid.")
            
        self.base_estimator_ = clone(self.base_estimator)
        
        # Define an objective function according to the chosen criterion
        @spy
        def _negative_objective(params):
            # Train the classifier
            self.__train(X, y, sensitive, params)
            
            # Compute the goal
            y_pred = self.base_estimator_.predict(X)
            objective = self.criterion(y, y_pred, sensitive)
            
            return -objective
            
        # Perform optimization
        # TODO: Pass optimization parameters as classifier parameters
        bounds = [(0, 1), (0, 1), (0, 3), (0, 3)]
        x0 = np.array([np.random.uniform(*b) for b in bounds])
        cb = CallbackCollector(_negative_objective, threshold=1e-2)
        try:
            res = minimize(
                _negative_objective,
                x0=x0,
                method="Powell",
                bounds=bounds,
                callback=cb,
            )
            best_x = res.x
        except (KeyboardInterrupt, StopIteration) as e:
            best_x = cb._last_x
        finally:
            self.best_params_ = best_x

        # ~ if not res.success:
            # ~ try:
                # ~ msg = res.message.decode("utf-8")
            # ~ except AttributeError:
                # ~ msg = res.message
            # ~ raise RuntimeError(
                # ~ "Optimization failed to converge:\n"
                # ~ + msg
            # ~ )
        
        # Retrain the classifier with the found best parameters
        # TODO: Recover the best screened values of parameters
        # Currently, self.best_params_ stores the last found vector,
        # which is not necessarily the best one.
        # ~ self.__train(X, y, sensitive, self.best_params_)
        
        return self
        
    def __train(self, X, y, sensitive, params):
        # Initialize weights
        m = len(X)
        w = np.ones(m)
        w_prev = w.copy() + np.sqrt(self.eps)
        
        # Extract parameters
        a_s, a_ns, b_s, b_ns = params
        
        while np.linalg.norm(w-w_prev) >= self.eps:
            
            # Train the classifier with normalized weights
            self.base_estimator_.fit(X, y, sample_weight=w)
            
            # Make a prediction
            y_pred = self.base_estimator_.predict_proba(X)[:,1]
            
            # Compute sensitive and non-sensitive errors          
            e_s = y_pred[sensitive] - y[sensitive]
            e_ns = y_pred[~sensitive] - y[~sensitive]
            
            # Update weights
            w_prev = w.copy()
            w[sensitive] = self._loss(e_s, b_s) * a_s \
                           + self._loss(-e_s, b_s) * (1 - a_s)
            w[~sensitive] = self._loss(e_ns, b_ns) * (1 - a_ns) \
                            + self._loss(-e_ns, b_ns) * a_ns
            w = w / sum(w) * len(w)
    
    @if_delegate_has_method(delegate='base_estimator_')
    def predict(self, X):
        """Predict classes for X.
        
        The predicted class of an input sample is computed according to
        the base estimator that underwent the optimization procedure.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        
        return self.base_estimator_.predict(X)
        
    def _more_tags(self):
        tags = {
            "binary_only": True,
        }
        
        return tags
