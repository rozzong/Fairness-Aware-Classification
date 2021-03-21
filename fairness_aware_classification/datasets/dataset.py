# -*- coding: utf-8 -*-
"""
The base dataset class.
"""

from abc import ABC, abstractmethod
import os
import warnings 

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from ..utils import sensitive_mask_from_features


class Dataset(ABC):
    
    # Define raw and preprocessed data paths
    RAW_DATA_PATH = \
        os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "raw_data"
    CLEAN_DATA_PATH = \
        os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "data"
        
    RAW_FILENAME = None
    CLEAN_FILENAME = None
    PARSE_KWARGS = None
    NA_VALUES = None
    FEATURES_TO_DROP = None
    FEATURES_TO_KEEP = None
    FORCE_NUM = None
    FORCE_CAT = None
    TARGET_LABEL = None
    SENSITIVE_FEATURES = None
    SENSITIVE_VALUES = None
        
    def __init__(self):
        
        # Check for the presence of the pre-processed file
        file_path = self.CLEAN_DATA_PATH + os.sep + self.CLEAN_FILENAME
        is_prepared = os.path.isfile(file_path)
        
        if is_prepared:
            # Load the prepared data
            self._df = pd.read_csv(file_path, index_col=0)
            
        else:
            # Load the raw data
            file_path = self.RAW_DATA_PATH + os.sep + self.RAW_FILENAME
            self._df = pd.read_csv(file_path, **self.PARSE_KWARGS)
            
            # Remove entries with unknown values
            if self.NA_VALUES:
                for v in self.NA_VALUES:
                    self._df.replace(v, np.nan, inplace=True)
                self._df.dropna(inplace=True)
                
            # Do some custom operation
            self._custom()
                
            # Drop some features
            if self.FEATURES_TO_DROP:
                if self.TARGET_LABEL not in self.FEATURES_TO_DROP:
                    self._df.drop(self.FEATURES_TO_DROP, axis=1)
                else:
                    raise ValueError("The target label '" \
                        + self.TARGET_LABEL + "' cannot be dropped")
            
            # Keep only features of interest and the target
            if self.FEATURES_TO_KEEP:
                if self.TARGET_LABEL in self.FEATURES_TO_KEEP:
                    self.FEATURES_TO_KEEP.remove(self.TARGET_LABEL)
                self._df = self._df[self.FEATURES_TO_KEEP + [self.TARGET_LABEL]]
            
            # Save index and columns
            self.index, self.columns = self._df.index, self._df.columns
            
            # Change some types, if asked
            if self.FORCE_NUM:
                self._df = self._df.astype({label: float for label in self.FORCE_NUM})
            if self.FORCE_CAT:
                self._df = self._df.astype({label: object for label in self.FORCE_CAT})
            
            # Get numerical and categorical features
            self.num_features = make_column_selector(dtype_include=np.number)(self._df)
            self.cat_features = make_column_selector(dtype_include=object)(self._df)
            
            # Start preprocessing
            self._prepare()
            
            # Rebuild the dataframe
            df = pd.DataFrame(index=self.index, columns=self.columns)
            try:
                df[self.num_features] = self.df_num_array
            except e:
                warnings.warn("No operation was done on numerical features.")
            try:
                df[self.cat_features] = self.df_cat_array
            except e:
                warnings.warn("No encoding was done on categorical features.")
            
            # Reapply correct dtypes
            for f in self.columns:
                if f in self.num_features:
                    df[f] = df[f].astype(np.float64)
                elif f in self.cat_features:
                    df[f] = df[f].astype(np.int64)
                    
            # Save the dataset
            self._df = df
            self._df.to_csv(
                self.CLEAN_DATA_PATH + os.sep + self.CLEAN_FILENAME,
            )
            
        # Set the target and do some feature selection
        self.y = self._df.pop(self.TARGET_LABEL) 
        self.X = self._df.copy()
        del self._df
        
        # Set default sensitive attributes
        self.sensitive = sensitive_mask_from_features(
            self.X,
            self.SENSITIVE_FEATURES,
            self.SENSITIVE_VALUES,
        )
        
    @abstractmethod
    def _custom(self):
        pass
    
    @abstractmethod
    def _prepare(self):
        pass
    
    def _scale(self):
        """
        Scale the numerical features.
        """
        scaler = make_column_transformer(
            (StandardScaler(), self.num_features),
            remainder="drop"
        )
        self.df_num_array = scaler.fit_transform(self._df)
        
    def _encode_ordinal(self):
        """
        Encode categorical features ordinally.
        """
        enc = make_column_transformer(
            (OrdinalEncoder(), self.cat_features),
            remainder="drop"
        )
        self.df_cat_array = enc.fit_transform(self._df)
    
    def _encode_one_hot(self):
        """
        Encode categorical features with a one hot scheme.
        """
        enc = make_column_transformer(
            (OneHotEncoder(drop='if_binary', sparse=False), self.cat_features),
            remainder="drop"
        )
        self.df_cat_array = enc.fit_transform(self._df)
        
        # Rebuild encoded columns
        # TODO: Make that less cryptic
        was_encoded = enc.transformers_[0][1].drop_idx_ == None
        i_encoded = np.where(was_encoded)[0]
        i_not_encoded = np.where(~was_encoded)[0]
        
        new_columns = []
        new_cat_features = []
        cat_encoded_features = []
        cat_not_encoded_features = []

        for feature in self.columns:
            # If the feature was among to-be-encoded candidates
            if feature in enc.transformers_[0][2]:
                i = enc.transformers_[0][2].index(feature)
                # If the feature actually was encoded
                if i in i_encoded:
                    new_features = enc.transformers_[0][1].categories_[i]
                    new_columns.extend(list(new_features))
                    new_cat_features.extend(list(new_features))
                    cat_encoded_features.extend(list(new_features))
                else:
                   new_columns.append(feature)
                   new_cat_features.append(feature)
                   cat_not_encoded_features.append(feature)
            else:
                new_columns.append(feature)
        
        # Update attributes        
        self.columns = new_columns
        self.cat_features = new_cat_features
                
    @abstractmethod
    def objective(self, y_true, y_pred, sensitive):
        """Objective function.
        
        This objective function is meant to be maximized, and can guide
        an optimization process to increase fairness.
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            Ground truth (correct) labels.
            
        y_pred : 1d array-like, or label indicator array / sparse matrix
            Predicted labels, as returned by a classifier.
            
        sensitive : array-like of shape (n_samples,)
            Mask indicating which samples are sensitive. The array should
            contain boolean values, where True indicates that the corresponding
            sample is sensitive.
            
        Returns
        -------
        score : float
            The score.
        """
        raise NotImplementedError("No objective function was set for " \
            + type(self) + ".")
