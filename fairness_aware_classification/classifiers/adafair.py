# -*- coding: utf-8 -*-
"""
The implementation of AdaFair.
"""

# Authors: Bekarys Nurtay, Gabriel Rozzonelli
# Based on the work of of the following paper:
# [1] V. Iosifidis, et E. Ntoutsi, « AdaFair: Cumulative Fairness
#     Adaptive Boosting ».

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier#, clone
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..metrics import dfpr_score, dfnr_score, ber_score, er_score, eq_odds_score


def ada_fair_alpha(y_true, y_pred_t, distribution):
    """
    Compute the weight for a current base estimator.
    
    Parameters:
    -----------
    y_true: 1-D array
        The true target values.
        
    y_pred: 1-D array
        The predicted values.
        
    distribution: 1-D array
        The weights of training instances.
    
    Returns:
    --------
    alpha: float
        The weight for current base estimator.
    """
    n = ((y_true != y_pred_t) * distribution).sum() / distribution.sum()
    alpha = np.log((1-n)/n) / 2

    return alpha
  
def ada_fairness_cost(y_true, y_pred, sensitive, eps):
    """
    Compute the fairness cost for sensitive features.
    
    Parameters:
    -----------
    y_true: 1-D array
        The true target values.
    
    y_pred: 1-D array
        The predicted values.
        
    sensitive : array-like of shape (n_samples,)
        Mask indicating which samples are sensitive. The array should
        contain boolean values, where True indicates that the corresponding
        sample is sensitive.
        
    eps: float
        The error threshold.
        
    Returns:
    --------
    u: 1-D array
        The fairness cost for each instance.
    """
    u = np.zeros((len(y_true),))
    
    # Compute criteria
    protected_pos = ((y_true == 1) & ~sensitive).astype(int)
    unprotected_pos = ((y_true == 1) & sensitive).astype(int)
    protected_neg= ((y_true == -1) & ~sensitive).astype(int)
    unprotected_neg = ((y_true == -1) & sensitive).astype(int)
    
    dfpr = dfpr_score(y_true, y_pred, sensitive)
    dfnr = dfnr_score(y_true, y_pred, sensitive)
    
    if abs(dfnr) > eps:
        if dfnr > 0:
            u[protected_pos & (y_true[protected_pos] != y_pred[protected_pos])] = abs(dfnr) 
        elif dfnr < 0:
            u[unprotected_pos & (y_true[unprotected_pos] != y_pred[unprotected_pos])] = abs(dfnr) 
    if abs(dfpr) > eps:
        if dfpr > 0:
            u[protected_neg & (y_true[protected_neg] != y_pred[protected_neg])] = abs(dfpr) 
        elif dfpr < 0:
            u[unprotected_neg & (y_true[unprotected_neg] != y_pred[unprotected_neg])] = abs(dfpr) 
    
    return u

def ada_fair_distribution(y_true, y_pred_t, conf_score, distribution, alpha_t, u_t, z_t=1.):
    """
    Compute weights of data instances for the next base estimator.
    
    Parameters:
    -----------
    y_true: 1-D array
        The true target values.
    
    y_pred: 1-D array
        The predicted values.
        
    conf_score: 1-D array
        The current base estimator's prediction confidence, in range [-1,1],
        where -1 is the strongest confidence on the misclassified class
        and 1 the strongest confidence on the correctly classified class.
    
    distribution: 1-D array
        The previous instance weights.
        
    alpha_t: float
        The current base estimator weight.
        
    u_t: 1-D array
        The current prediction fairness cost.
        
    z_t: float, default=1.
        The normalization factor.
        
    Returns:
    --------
    distribution: 1-D array
        The updated instance weights.
    """
    distribution = 1/z_t * distribution \
        * np.exp(alpha_t*conf_score*(y_true!=y_pred_t)) * (1+u_t)
    
    return distribution

class AdaFairClassifier(BaseEstimator, ClassifierMixin):
    """AdaFair Classifier (AdaFairClassifier)
    
    Parameters
    ----------
    n_estimators: int, default=50
        The number of base estimators.
        
    base_classifier: object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required.
        
    eps: float, default=1e-3
        The error threshold.
        
    c: float
        The balancing coefficient for number of base classifier optimizer.
        It should be in the range [0, 1].
        
    get_alpha: function, default=None
        The function used to calculate and return alpha. If set to None,
        the default alpha computation method is applied.
        
    update_distribution: function, default=None
        The updates weights distribution. If set to None, the default
        weights computation method is applied.
        
    get_fairness_cost: function, default=None
        The fairness cost for predictions. If set to None, the default
        cost function is used.
        
    Attributes
    ----------        
    optimum_: int
        The optimal number of base estimators.
        
    alphas_: list
        The weights of base estimators.
        
    classifiers_: list
        The base estimators.
        
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
        
    n_features_in_: int
        The number of features the classifier was fit for.
    """
    def __init__(self, n_estimators=50, base_classifier=None, eps=1e-3, c=1,
                 get_alpha=None, update_distribution=None,
                 get_fairness_cost=None):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.eps = eps
        self.c = c
        self.get_alpha = get_alpha
        self.update_distribution = update_distribution
        self.get_fairness_cost = get_fairness_cost
        
    def fit(self, X, y, sensitive=None):
        """Build an AdaFair classifier from the training set (X, y) and
        known sensitive samples.
        
        Parameters:
        -----------
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
        
        if not isinstance(sensitive, np.ndarray):
            if isinstance(sensitive, pd.Series):
                sensitive = sensitive.to_numpy()
            else:
                raise ValueError("Argument 'sensitive' is not valid.")
        
        # Check that the base estimator is an classifier
        if not is_classifier(self.base_classifier()):
            raise ValueError("The base estimator should be a classifier, " \
                             "but it is not.")
        
        # Check that the it is a binary classification problem
        check_classification_targets(y)
        n_classes = len(np.unique(y))
        if type_of_target(y) != "binary":
            raise ValueError("Target values should belong to a binary set, " \
                             "but {} classes were found.".format(n_classes))
        
        # TODO: Instanciate new attributes for those methods, as self._x
        if self.get_alpha is None:
            self.get_alpha = ada_fair_alpha
        elif not callable(self.get_alpha):
            raise ValueError("Argument 'get_alpha' is not valid.")
            
        if self.update_distribution is None:
            self.update_distribution = ada_fair_distribution
        elif not callable(self.update_distribution):
            raise ValueError("Argument 'update_distribution' is not valid.")
            
        if self.get_fairness_cost is None:
            self.get_fairness_cost = ada_fairness_cost
        elif not callable(self.get_fairness_cost):
            raise ValueError("Argument 'get_fairness_cost' is not valid.")
            
        # Handle the case with no specified sensitive sample
        if sensitive is None:
            warnings.warn("Argument `sensitive` is not set. " \
                          "Setting all samples as non-sensitives.")
            sensitive = np.zeros(len(y)).astype(bool)
        
        # Save number of features
        self.n_features_in_ = X.shape[1]
        
        # Save classes
        self.classes_, y = np.unique(y, return_inverse=True)
        
        # Readjust y to have it in {-1, 1}
        y_ada = (2 * y - 1).astype(int)
        
        self.optimum_ = 1
        self.classifiers_ = []
        self.alphas_ = []
        
        n_samples = X.shape[0]
        distribution = np.ones(n_samples, dtype=float) / n_samples
        min_err = np.inf
        y_preds = 0
            
        for i in range(self.n_estimators):
            
            # Train the last base classifier
            self.classifiers_.append(self.base_classifier())     
            self.classifiers_[-1].fit(X, y_ada, sample_weight=distribution)
            
            # Get predictions and prediction probabilities of the last classifier
            y_pred = self.classifiers_[-1].predict(X)
            y_proba = self.classifiers_[-1].predict_proba(X)[:,1]
            
            # Compute the confidence score derived from prediction probabilities
            confidence = (y_proba - 0.5) / 0.5 * y_pred
            
            # Compute the weight of the current classifier 
            alpha_t = self.get_alpha(y_ada, y_pred, distribution)
            self.alphas_.append(alpha_t)
            
            # Update of weighted votes of all fitted base estimators
            y_preds += (y_pred * alpha_t)
            
            # Compute the fairness cost for the current base learner predictions
            u = self.get_fairness_cost(y_ada, y_pred, sensitive, self.eps)
            
            # Update weights of instances
            distribution = self.update_distribution(
                y_ada, y_pred, confidence, distribution, alpha_t, u
            )
            
            # Get the sign of the weighted predictions
            y_preds_s = np.sign(y_preds)
            
            # Put it back in {0, 1}
            y_preds_s = (1 + y_preds_s) / 2
            
            # Find the optimal number of base classifiers, as the
            # minimum of the sum of BER, ER and Eq.Odds scores
            curr_err = self.c * ber_score(y, y_preds_s) \
                       + (1-self.c) * er_score(y, y_preds_s) \
                       + eq_odds_score(y, y_preds_s, sensitive)
            
            if min_err > curr_err:
                min_err = curr_err
                self.optimum_ = i + 1
                
        return self

           
    def predict(self, X, end=None):
        """Predict classes for X.
        
        The predicted class of an input sample is computed by weighted
        voting of base estimators.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        end: int or str, default=None
            The number of base estimators to use to do the prediction.
            
            - If set to None, all the trained estimators are used.
            - If set to an integer value `end`, the first `end` estimators
              are used.
            - If set to `optimum`, an optimum number of estimators
              determined during training is used.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        
        if end == "optimum":
            end = self.optimum_
            
        final_predictions = np.zeros(X.shape[0])

        for alpha, h in zip(self.alphas_[:end], self.classifiers_[:end]):
            final_predictions += alpha * h.predict(X)
        
        out = np.sign(final_predictions)
        
        # Readjust the signed predictions to be in {0, 1}
        out = ((1 + out) / 2).astype(int)

        return self.classes_[out]
        
    def _more_tags(self):
        tags = {
            "binary_only": True,
        }
        
        return tags
