# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_aware_classification.adaptive_weights import AdaptiveWeightsClassifier
from fairness_aware_classification.metrics import dfpr, dfnr
from fairness_aware_classification.utils import sensitive_mask_from_features


def objective(y_true, y_pred, sensitive_samples):
    acc = accuracy_score(y_true, y_pred)
    d_p = dfpr(y_true, y_pred, sensitive_samples)
    d_n = dfnr(y_true, y_pred, sensitive_samples)
    
    return 2 * acc - abs(d_p) - abs(d_n)
    

if __name__ == "__main__":
    # Load and split the data
    df = pd.read_csv("data/propublica_data_for_fairml.csv")
    
    # Set the target and do some feature selection
    y = df.pop("Two_yr_Recidivism")
    X = df.drop(["score_factor", "Asian", "Hispanic", "Native_American", "Other"], axis=1)
    
    # Compute the sensitive samples mask
    sensitive_features = ["Female", "African_American"]
    sensitive = sensitive_mask_from_features(X, sensitive_features)
    
    # Split the data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, sensitive, test_size=0.4)
    
    # Classifier to reweight
    base_clf = LogisticRegression(penalty='none')
    
    # The criterion function `objective` should be customized
    # depending on the data. It should be maximized.
    awc = AdaptiveWeightsClassifier(base_clf, objective)
    
    # Get predictions of the base classifier
    base_clf.fit(X_train, y_train)
    y_pred_base = base_clf.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    obj_base = objective(y_test, y_pred_base, s_test)
    
    # Get prediction for the weighted classsifier
    awc.fit(X_train, y_train, s_train)
    y_pred_weighted = awc.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    obj_weighted = objective(y_test, y_pred_weighted, s_test)
    
    # Print results
    print("{}:".format(type(base_clf).__name__))
    print("* Accuracy: {:.4f}".format(acc_base))
    print("* Objective: {:.4f}".format(obj_base))
    print("AdaptiveWeightsClassifier")
    print("* Accuracy: {:.4f}".format(acc_weighted))
    print("* Objective: {:.4f}".format(obj_weighted))

