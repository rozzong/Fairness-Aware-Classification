# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_aware_classification.adaptive_weights import AdaptiveWeightsClassifier
from fairness_aware_classification.metrics import dfpr, dfnr, p_rule
from fairness_aware_classification.datasets import KDDDataset
from fairness_aware_classification.utils import sensitive_mask_from_features
    

if __name__ == "__main__":
    # Load the data
    data = KDDDataset()
    
    # Split the data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        data.X,
        data.y,
        data.sensitive,
        test_size=0.3,
    )
    
    # Classifier to reweight
    base_clf = LogisticRegression(solver="liblinear")
    
    # The criterion function `objective` should be customized
    # depending on the data. It should be maximized.
    awc = AdaptiveWeightsClassifier(base_clf, data.objective)
    
    # Get predictions of the base classifier
    base_clf.fit(X_train, y_train)
    y_pred_base = base_clf.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    obj_base = data.objective(y_test, y_pred_base, s_test)
    
    # Get prediction for the weighted classsifier
    awc.fit(X_train, y_train, s_train)
    y_pred_weighted = awc.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    obj_weighted = data.objective(y_test, y_pred_weighted, s_test)
    
    # Print results
    print("{}:".format(type(base_clf).__name__))
    print("* Accuracy: {:.4f}".format(acc_base))
    print("* Objective: {:.4f}".format(obj_base))
    print("AdaptiveWeightsClassifier")
    print("* Accuracy: {:.4f}".format(acc_weighted))
    print("* Objective: {:.4f}".format(obj_weighted))
