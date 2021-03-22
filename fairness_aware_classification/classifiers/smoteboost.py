# -*- coding: utf-8 -*-
"""
The implementation of SMOTEBoost.
"""

# Authors: Simona Nitti, Gabriel Rozzonelli
# Based on the work of of the following paper:
# [1] N. Chawla, A. Lazarevic, L. Hall, et K. Bowyer, « SMOTEBoost: 
#     Improving Prediction  of the Minority Class in Boosting ».

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import is_classifier, clone
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class SMOTE:
    """SMOTE
    
    Performs SMOTE resampling to address class imbalance.
    
    Parameters
    ----------
    k_neighbors : int, default=5
        The number of nearest neighbors.
        
    Attributes
    ----------
    k_neighbors_ : int
        The number of nearest neighbors.
    """

    def __init__(self, k_neighbors=5):
        self.k_neighbors_ = k_neighbors

    def fit(self, X):
        """Fit SMOTE on a training set, by looking for the `k_neighbors`
        nearest neighbors of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           The samples to oversample from.
        """
        self.X = check_array(X)
        self.n_features_in_ = self.X.shape[1]

        # Fit nearest neighbors
        n_neighbors = self.k_neighbors_ + 1
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors)
        self.neigh.fit(self.X)

        return self

    def sample(self, n_samples):
        """
        Generate new synthetic samples from the training samples.

        Parameters
        ----------
        n_samples : int
            The number of new synthetic samples to generate.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
            The new synthetic samples.
        """
        X_new = np.zeros((n_samples, self.n_features_in_))
        
        for i in range(n_samples):
            
            # Pick a sample randomly
            j = np.random.choice(range(self.X.shape[0]))

            # Take the k nearest neighbors around it
            X_j = self.X[j].reshape(1, -1)
            new_neighs = self.neigh.kneighbors(X_j, return_distance=False)
            
            # Keep all columns but the first one as it is X[j] itself
            new_neighs = new_neighs[:,1:]
            
            # Choose one of the k neighbors
            new_neigh_index = np.random.choice(new_neighs[0])  
            
            # Measure the index between X[j] and the randomly chosen neighbor
            distance = self.X[new_neigh_index] - self.X[j] 
            fraction = np.random.random()
            
            # Synthetize a new sample
            X_new[i] = self.X[j] + fraction * distance

        return X_new

        
class SMOTEBoostClassifier(BaseEstimator, ClassifierMixin):
    """SMOTEBoost Classifier (SMOTEBoostClassifier)
    
    Parameters
    ----------        
    base_classifier : object
        Base classifier from which the boosted ensemble is built.
        
    n_estimators : int, default=3
        The number of base estimators.
        
    k_neighbors : int, default=5
        Number of nearest neighbors for SMOTE.
        
    n : int, default=5
        The number of new synthetic samples per boost iteration.
        
    Attributes
    ----------
    classifiers_ : list
        The collection of fitted base classifiers.

    classes_ : array of shape (n_classes,)
        The classes labels.
    
    minority_class_ : int
        Class identified as the minority class.

    alphas_ : array of shape (n_estimators,)
        The weights for each estimator in the boosted ensemble.
    """
    
    _required_parameters = ["base_classifier"]

    def __init__(self, base_classifier, n_estimators=3, k_neighbors=5, n=5): 
        self.base_classifier = base_classifier  
        self.n_estimators = n_estimators  
        self.k_neighbors = k_neighbors
        self.n = n

    def fit(self, X, y):
        """Build a SMOTEBoost classifier from the training set (X, y).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
           The training input samples.

        y : array-like of shape (n_samples,)
           The target values (class labels) as integers.

        Returns:
        --------
        self: object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        # Check that the base estimator is a classifier
        if not is_classifier(self.base_classifier):
            raise ValueError("The base estimator should be a classifier, " \
                             "but it is not.")
                             
        # Check that the it is a binary classification problem
        check_classification_targets(y)
        n_classes = len(np.unique(y))
        if type_of_target(y) != "binary":
            raise ValueError("Target values should belong to a binary set, " \
                             "but {} classes were found.".format(n_classes))
        
        # Initialize lists to hold models and model weights
        self.classifiers_ = []
        self.alphas_ = []

        # Find the minority class
        self.classes_, counts = np.unique(y, return_counts=True)
        self.minority_class_ = self.classes_[counts==-np.max(-counts)][0]
        X_minority = X[y == self.minority_class_]
        
        # Fit SMOTE on the sensitive samples       
        smote = SMOTE(k_neighbors=self.k_neighbors)
        smote.fit(X_minority)

        # Initialize the distribution
        dist = np.ones_like(X) / (len(y) * (n_classes-1))  
        for i in range(len(y)):
            dist[i, np.where(self.classes_ == y[i])[0][0]] = 0

        for i in range(self.n_estimators):
            # Create new artificial samples from the minority class
            X_new = smote.sample(self.n)
            y_new = np.ones(self.n) * self.minority_class_

            # Append the new examples
            X_smote = np.concatenate((X, X_new))
            y_smote = np.concatenate((y, y_new))

            # Train a weak learner with the modified distribution
            self.classifiers_.append(clone(self.base_classifier))     
            self.classifiers_[i].fit(X_smote, y_smote)

            # Make predictions over the initial dataset
            h = self.classifiers_[i].predict_proba(X) 

            # Compute the pseudo-loss of hypothesis
            # TODO: Try to avoid nested loops
            epsilon = 0
            for k in range(len(y)):
                for j in range(n_classes):
                    epsilon += dist[k,j] \
                    * (1.- h[k,np.where(self.classes_ == y[k])[0][0]] + h[k,j])
            beta = epsilon / (1. - epsilon)
            self.alphas_.append(np.log(1/beta))

            # Update distribution
            # TODO: Try to avoid nested loops
            z = np.sum(dist)
            for k in range(len(y)):
                for j in range(n_classes):
                    exp = (1. + h[k,np.where(self.classes_==y[k])[0][0]] - h[k,j]) / 2
                    dist[k,j] = dist[k,j] / z * beta**exp
    
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
        
        X = check_array(X)
        
        final_predictions = np.zeros((X.shape[0], len(self.classes_)))
        y = np.zeros(X.shape[0])

        for t in range(self.n_estimators):
            final_predictions \
                += self.classifiers_[t].predict_proba(X) * self.alphas_[t]
                
        for i in range(len(X)):
            y[i] = self.classes_[final_predictions[i,:] == np.amax(final_predictions[i,:])][0]
        
        return y
        
    def _more_tags(self):
        tags = {
            "binary_only": True,
        }
        
        return tags
