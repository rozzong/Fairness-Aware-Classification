# -*- coding: utf-8 -*-

import warnings 

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import scipydirect as direct


class AdaptiveWeightsClassifier(BaseEstimator, ClassifierMixin):
    """
    Adapative Weights Classifier (AdaptiveWeightsClassifier)
    
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
    """
    
    def __init__(self, base_estimator, criterion, loss="exp", eps=0.1):
        self.base_estimator = base_estimator
        self.criterion = criterion            
        self.loss = loss
        self.eps = eps
        
    def fit(self, X, y, sensitive_samples=None):
        """Build a weighted classifier from the training set (X, y) and
        known sensitive samples.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
            
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        sensitive_samples : array-like of shape (n_sensitive_samples,), default:None
            Indices of the sensitive samples. The array should be filled
            with intergers. If set to None, it is assumed that no sample
            is sensitive.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Perform checks
        X, y = check_X_y(X, y)
        assert is_classifier(self.base_estimator)
        
        if sensitive_samples is None:
            warnings.warn("Argument `sensitive_samples` is not set.")
            sensitive_samples = np.arange(0)
        
        # Update the convex loss function if needed, or check it
        if self.loss is "exp":
            self.loss = lambda e, b: np.exp(b*e)
        else:
            if not callable(self.loss):
                raise ValueError("Argument 'loss' is not valid.")
            
        self.base_estimator_ = clone(self.base_estimator)
        
        # Define an objective function according to the chosen criterion
        def _negative_objective(params):
            # Train the classifier
            self.__train(X, y, sensitive_samples, params)
            
            # Compute the goal
            y_pred = self.base_estimator_.predict(X)
            objective = self.criterion(y, y_pred, sensitive_samples)
            
            return -objective
            
        # Perform DIRECT optimization
        # TODO: Pass optimization parameters as classifier parameters
        bounds = [(0, 1), (0, 1), (0, 3), (0, 3)]
        res = direct.minimize(
            _negative_objective,
            bounds=bounds,
            maxF=320,
            maxT=80,
        )
        
        if not res.success:
            raise RuntimeError(
                "DIRECT optimization failed to converge:\n"
                + res.message
            )
        
        self.best_params_ = res.x
        
        # Retrain the classifier with the found best parameters
        # TODO: Check if it is really useful
        self.__train(X, y, sensitive_samples, self.best_params_)
        
        return self
        
    def __train(self, X, y, sensitive_samples, params):
        # Initialize weights
        m = len(X)
        w = np.ones(m)
        w_prev = w.copy() + np.sqrt(self.eps)
        
        # Build a mask
        sensitive = np.isin(range(m), sensitive_samples)
        
        # Extract parameters
        a_s, a_ns, b_s, b_ns = params
        
        while np.sum((w-w_prev)**2) >= self.eps:
            
            # Train the classifier with normalized weights
            w_normalized = w / np.sum(w)
            self.base_estimator_.fit(X, y, sample_weight=w_normalized)
            
            # Make a prediction
            y_pred = self.base_estimator_.predict(X)
            
            # Compute sensitive and non-sensitive errors          
            e_s = y_pred[sensitive] - y[sensitive]
            e_ns = y_pred[~sensitive] - y[~sensitive]
            
            # Update weights
            w_prev = w
            w[sensitive] = self.loss(e_s, b_s) * a_s \
                           + self.loss(-e_s, b_s) * (1 - a_s)
            w[~sensitive] = self.loss(e_ns, b_ns) * (1 - a_ns) \
                            + self.loss(-e_ns, b_ns) * a_ns
    
    def predict(self, X):
        """Predict classes for X.
        
        The predicted class of an input sample is computed according to
        the base estimator that underwent the optimization procedure.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        # Perform checks
        check_is_fitted(self)
        X = check_array(X)
        
        # Perform the prediction
        y = self.base_estimator_.predict(X)
        
        return y
