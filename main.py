# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_aware_classification.classifiers import AdaFairClassifier, \
    AdaptiveWeightsClassifier, SMOTEBoostClassifier
from fairness_aware_classification.datasets import *
from fairness_aware_classification.utils import sensitive_mask_from_features


if __name__ == "__main__":
                
    # Load the data
    data = COMPASDataset()
    dataset_name = type(data).__name__
    
    # Split the data
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        data.X,
        data.y,
        data.sensitive,
        test_size=0.5,
    )
    
    # Select base classfiers for the meta-classifiers
    base_clf_af = lambda: DecisionTreeClassifier(max_depth=2)
    base_clf_aw = LogisticRegression(solver="liblinear")
    base_clf_sb = lambda: LogisticRegression()
    
    # Single base classifier for the sake of comparison
    base_clf = clone(base_clf_aw)
    base_clf_name = type(base_clf).__name__
    
    # The criterion function `objective` should be customized
    # depending on the data. It should be maximized.
    af = AdaFairClassifier(50, base_clf_af)
    aw = AdaptiveWeightsClassifier(base_clf, data.objective)
    sb = SMOTEBoostClassifier(base_clf_sb, 4, 20, 500)
    
    # Get predictions of the base classifier
    base_clf.fit(X_train, y_train)
    y_pred_base = base_clf.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    obj_base = data.objective(y_test, y_pred_base, s_test)
    
    # Get prediction for the AdaFair classifier
    af.fit(X_train, y_train, s_train)
    y_pred_af = af.predict(X_test)
    acc_af = accuracy_score(y_test, y_pred_af)
    obj_af = data.objective(y_test, y_pred_af, s_test)
    
    # Get prediction for the weighted classifier
    aw.fit(X_train, y_train, s_train)
    y_pred_aw = aw.predict(X_test)
    acc_aw = accuracy_score(y_test, y_pred_aw)
    obj_aw = data.objective(y_test, y_pred_aw, s_test)
    
    # Get prediction for the SMOTEBoost classifier
    sb.fit(X_train, y_train)
    y_pred_sb = sb.predict(X_test)
    acc_sb = accuracy_score(y_test, y_pred_sb)
    obj_sb = data.objective(y_test, y_pred_sb, s_test)
    
    print(y_pred_base)
    print('*'*10)
    print(y_pred_af)
    print('*'*10)
    print(y_pred_aw)
    print('*'*10)
    print(y_pred_sb)
    print('*'*10)
    
    # Print results
    print("{}:".format(base_clf_name))
    print("* Accuracy: {:.4f}".format(acc_base))
    print("* Objective: {:.4f}\n".format(obj_base))
    print("AdaFairClassifier")
    print("* Accuracy: {:.4f}".format(acc_af))
    print("* Objective: {:.4f}\n".format(obj_af))
    print("AdaptiveWeightsClassifier")
    print("* Accuracy: {:.4f}".format(acc_aw))
    print("* Objective: {:.4f}\n".format(obj_aw))
    print("SMOTEBoostClassifier")
    print("* Accuracy: {:.4f}".format(acc_sb))
    print("* Objective: {:.4f}\n".format(obj_sb))
    
    
