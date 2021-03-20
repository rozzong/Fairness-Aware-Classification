# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fairness_aware_classification.classifiers import AdaFairClassifier, AdaptiveWeightsClassifier
# ~ from fairness_aware_classification.ada_fair import AdaptiveWeightsClassifier
from fairness_aware_classification.datasets import *
from fairness_aware_classification.utils import sensitive_mask_from_features


if __name__ == "__main__":
    
    # ~ for Dataset in [KDDDataset]: #AdultDataset, BankDataset, COMPASDataset, 
        # ~ data = Dataset()
        # ~ print(Dataset.__name__)
        # ~ np.save(Dataset.__name__+"_sensitive_mask.npy", data.sensitive)
        # ~ np.save(Dataset.__name__+"_y.npy", data.y)
        # ~ np.save(Dataset.__name__+"_sensitive_value.npy", data.sensitive_values)
        # ~ np.save(Dataset.__name__+"_sensitive_mask.npy", data.sensitive)
    # ~ input()
    # ~ input("end")
            
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
    
    # Classifier to reweight
    # ~ base_clf = LogisticRegression(solver="liblinear")
    # ~ base_clf_name = type(base_clf).__name__
    base_clf = lambda: DecisionTreeClassifier(max_depth=2)
    base_clf_name = type(base_clf).__name__
    
    # The criterion function `objective` should be customized
    # depending on the data. It should be maximized.
    # ~ awc = AdaptiveWeightsClassifier(base_clf, data.objective)
    af = AdaFairClassifier(50, base_clf)
    
    # Get predictions of the base classifier
    # ~ base_clf.fit(X_train, y_train)
    # ~ y_pred_base = base_clf.predict(X_test)
    # ~ acc_base = accuracy_score(y_test, y_pred_base)
    # ~ obj_base = data.objective(y_test, y_pred_base, s_test)
    
    # Get prediction for the AdaFair classsifier
    af.fit(X_train, y_train, s_train)
    y_pred_af = af.predict(X_test)
    acc_af = accuracy_score(y_test, y_pred_af)
    obj_af = data.objective(y_test, y_pred_af, s_test)
    
    # Get prediction for the weighted classsifier
    # ~ awc.fit(X_train, y_train, s_train)
    # ~ y_pred_awc = awc.predict(X_test)
    # ~ acc_awc = accuracy_score(y_test, y_pred_awc)
    # ~ obj_awc = data.objective(y_test, y_pred_awc, s_test)
    
    # Print results
    # ~ print("{}:".format(base_clf_name))
    # ~ print("* Accuracy: {:.4f}".format(acc_base))
    # ~ print("* Objective: {:.4f}".format(obj_base))
    print("AdaFairClassifier")
    print("* Accuracy: {:.4f}".format(acc_af))
    print("* Objective: {:.4f}".format(obj_af))
    # ~ print("AdaptiveWeightsClassifier")
    # ~ print("* Accuracy: {:.4f}".format(acc_awc))
    # ~ print("* Objective: {:.4f}".format(obj_awc))
    
    # Save results
    # ~ np.save(dataset_name+"_awc_test.npy", y_test)
    # ~ np.save(dataset_name+"_awc_pred.npy", y_pred_awc)
    # ~ np.save(dataset_name+"_awc_mask.npy", s_test)
    
    
