# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from adaptive_weights import AdaptiveWeightsClassifier
from metrics import dfpr, dfnr


def objective(y_true, y_pred, sensitive_samples):
    # Compute the accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Compute the DFPR and DFNR
    d_p = dpnr(y_true, y_pred, sensitive_samples)
    d_n = dfnr(y_true, y_pred, sensitive_samples)
    
    return 2 * acc - abs(d_p) - abs(d_n)
    

if __name__ == "__main__":
    # Load and split the data
    # ~ data = None # TODO: Test with actual data
    # ~ X_train, X_test, y_train, y_test = train_test_split(data)
    
    # Classifier to reweight
    base_clf = LogisticRegression(penalty='none')
    
    # The criterion function `objective` should be customized
    # depending on the data. It should be maximized.
    awc = AdaptiveWeightsClassifier(base_clf, objective)
    
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(awc)
    
    # Get prediction for the weighted classsifier
    # ~ awc.fit(X_train, y_train, sensitive_samples)
    # ~ y_pred = awc.predict(X_test)
