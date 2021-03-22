# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from tqdm import tqdm

from fairness_aware_classification.classifiers import AdaFairClassifier, \
    AdaptiveWeightsClassifier, SMOTEBoostClassifier
from fairness_aware_classification.datasets import COMPASDataset
from fairness_aware_classification.utils import sensitive_mask_from_features


if __name__ == "__main__":
                
    # Load the data
    data = COMPASDataset()
    dataset_name = type(data).__name__
    
    # Select base classfiers for the meta-classifiers
    base_clf_af = DecisionTreeClassifier(max_depth=2)
    base_clf_aw = LogisticRegression(solver="liblinear")
    base_clf_sb = LogisticRegression()
    
    # Choose a single base classifier for the sake of comparison
    base_clf = clone(base_clf_aw)
    base_clf_name = type(base_clf).__name__
    
    # Define classifiers to test
    classifiers = {
        type(clf).__name__: clf for clf in [
            base_clf,
            AdaFairClassifier(base_clf_af, 50),
            AdaptiveWeightsClassifier(base_clf_aw, data.objective),
            SMOTEBoostClassifier(base_clf_sb, 4, 20, 500),
        ]
    }
    
    # Define scores
    get_scores = lambda y_test, y_pred, s_test: {
        "accuracy": accuracy_score(y_test, y_pred),
        "objective": data.objective(y_test, y_pred, s_test),
    }
    
    # Create a repeated stratified k-fold iterator
    rsfk = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)
    splits = rsfk.split(data.X, data.y)
    n_splits = rsfk.get_n_splits()
    
    # Create a dataframe to store the results
    columns = pd.MultiIndex.from_product(
        [list(classifiers), list(get_scores(*[np.array([0])]*3))],
        names=["Classifier", "Score"],
    )
    res = pd.DataFrame(index=range(n_splits), columns=columns)
    
    # Instanciate a progress bar
    pbar = tqdm(splits, total=n_splits)
    
    for i, (train_index, test_index) in enumerate(pbar):
        
        # Get the splits
        X_train, X_test = data.X.iloc[train_index], data.X.iloc[test_index]
        y_train, y_test = data.y.iloc[train_index], data.y.iloc[test_index]
        s_train, s_test = data.sensitive.iloc[train_index], data.sensitive.iloc[test_index]
        
        for clf_name, clf in classifiers.items():
            
            # Fit the classifier
            try:
                clf.fit(X_train, y_train, sensitive=s_train)
            except TypeError:
                clf.fit(X_train, y_train)
            
            # Make predictions and compute scores
            y_pred = clf.predict(X_test)
            scores = get_scores(y_test, y_pred, s_test)
            
            # Store scores
            for score, value in scores.items():
                res.loc[i, (clf_name, score)] = value
            
        pbar.update(1)
    
    # Save results
    res.to_csv("res.csv")
    
    # Import the results back
    # >>> res = pd.read_csv("res.csv", header=[0, 1])
    
