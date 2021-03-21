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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class SMOTE(object):
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

    def fit(self, X, y=None):
        """Fit SMOTE to the minority class, by looking for the `k_neighbors`
        nearest neighbors of each sample from the minority class.

        Parameters
        ----------
        X : array-like of shape (n_minority_samples, n_features)
            The samples from the minority class, already chosen.
            
        y : Ignored
            Not used, present here for API consistency by convention.
        """
        self.X_ = X
        self.n_minority_samples_, self.n_features_ = X.shape

        # Learn nearest neighbors
        # TODO: Why k + 1?
        n_neighbors = self.k_neighbors_ + 1
        self.neigh_ = NearestNeighbors(n_neighbors=n_neighbors)
        self.neigh_.fit(X)

        return self

    def sample(self, n_samples):
        """
        Generate new synthetic samples from the minority class.

        Parameters
        ----------
        n_samples : int
            The number of new synthetic samples to generate.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_features)
            The new synthetic samples.
        """
        X_new = np.zeros(shape=(n_samples, self.n_features_))
        
        # Calculate synthetic samples
        for i in range(n_samples):
            
            # Pick a sample randomly
            j = np.random.choice(range(self.X_.shape[0]))

            # Take the k nearest neighbors around it
            X_j_t = self.X_[j].reshape(1, -1)
            new_neighs = self.neigh_.kneighbors(X_j_t, return_distance=False)
            
            # Keep all columns but the first one as it is X[j] itself
            new_neighs = new_neighs[:,1:]
            
            # Choose one of the k neighbors
            new_neigh_index = np.random.choice(new_neighs[0])  
            
            # Measure the index between X[j] and the randomly choosen neighbor
            distance = self.X_[new_neigh_index] - self.X_[j] 
            fraction = np.random.random()
            
            # Synthetize a new sample by adding to X[j] a fraction of the distance
            # ~ X_new[i, :] = self.X[j, :] + fraction * distance[:]
            X_new[i] = self.X_[j] + fraction * distance

        return X_new

        
class SMOTEBoostClassifier(BaseEstimator, ClassifierMixin):
    """SMOTEBoost Classifier (SMOTEBoostClassifier)
    
    Parameters
    ----------        
    base_classifier : callable, default=None
        Callable that returns a base estimator from which the boosted
        ensemble is built.
        
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

    n_classes_ : int
        The number of classes.

    classes_ : array of shape (n_classes,)
        The classes labels.
    
    minority_class_ : int
        Class identified as the minority class.

    alphas_ : array of shape (n_estimators,)
        The weights for each estimator in the boosted ensemble.
    """

    def __init__(self, base_classifier=None, n_estimators=3, k_neighbors=5, n=5): 
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
        
        # Initialize lists to hold models and model weights (alphas)
        self.classifiers_ = []
        self.alphas_ = []

        # Find the minority class
        (unique, counts) = np.unique(y, return_counts=True)
        minority_class = unique[counts==-np.max(-counts)][0]
        X_minority = X[y == minority_class]
        self.n_classes_ = len(unique)
        self.classes_ = unique # List of y's possible classes
        self.minority_class_ = minority_class

        # Initialize the distribution D1 over the examples
        D = np.ones_like(X) / (len(y) * (self.n_classes_-1))  
        for i in range(len(y)):
            D[i, np.where(self.classes_ == y[i])[0][0]] = 0 

        # Iterate T times
        for i in range(self.n_estimators):
            # Modify the distribution by creating N synthetic examples from minority class
            smote = SMOTE(k_neighbors=self.k_neighbors)
            smote.fit(X_minority)
            X_new = smote.sample(self.n)
            y_new = np.ones(self.n) * minority_class

            # Append the new examples
            xsmote = np.concatenate((X, X_new))
            ysmote = np.concatenate((y, y_new))

            # Train a weak learner with the modified distribution
            self.classifiers_.append(self.base_classifier())     
            self.classifiers_[i].fit(xsmote, ysmote)

            # Make predictions over the initial dataset
            h = self.classifiers_[i].predict_proba(X) 

            # Compute the pseudo-loss of hypothesis ht
            epsilon = 0
            for k in range(len(y)):
                for j in range(self.n_classes_):
                    epsilon += D[k,j] \
                    * (1.- h[k,np.where(self.classes_ == y[k])[0][0]] + h[k,j])
            beta = epsilon / (1.-epsilon)
            self.alphas_.append(np.log(1/beta))

            # Update distribution
            Z = np.sum(D)
            for k in range(len(y)):
                for j in range(self.n_classes_):
                    D[k,j] = D[k,j] / Z * beta**(0.5*(1. + h[k,np.where(self.classes_==y[k])[0][0]] - h[k,j]))
    
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
        final_predictions = np.zeros((X.shape[0], self.n_classes_))
        y = np.zeros(X.shape[0])

        for t in range(self.n_estimators):
            final_predictions += self.classifiers_[t].predict_proba(X) * self.alphas_[t]
        for i in range(len(X)):
            y[i] = self.classes_[final_predictions[i,:] == np.amax(final_predictions[i,:])][0]
        
        return y
