# -*- coding: utf-8 -*-
"""
The KDD census income dataset.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from ..metrics import p_rule
from ..utils import sensitive_mask_from_features


RAW_DATA_PATH = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "raw_data"
DATA_PATH = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "data"

class KDDDataset:
    
    def __init__(self):
        # Try to load the preprocessed dump
        file_exists = os.path.isfile(DATA_PATH + os.sep + \
            "kdd_census_income_preprocessed.csv")
            
        if not file_exists:
            # Load and split the data
            file_path = RAW_DATA_PATH + os.sep + "kdd_census_income.csv"
            df = pd.read_csv(file_path, sep=", ", engine="python")
            
            # Remove entries with unknown values
            # That is a lot of loss, but there are still enough samples
            df.replace("?", np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Force some columns to be categorical
            df = df.astype({"industry code": object})
            df = df.astype({"occupation code": object})
            df = df.astype({"own business or self employed": object})
            
            # Save index and columns
            index, columns = df.index, df.columns
            
            # Get numerical and categorical features
            num_features = make_column_selector(dtype_include=np.number)(df)
            cat_features = make_column_selector(dtype_include=object)(df)
            
            # Perform scaling
            scaler = make_column_transformer(
                (StandardScaler(), num_features),
                remainder="drop"
            )
            df_num_array = scaler.fit_transform(df)
            
            # Perform encoding
            enc = make_column_transformer(
                (OneHotEncoder(drop='if_binary', sparse=False), cat_features),
                remainder="drop"
            )
            df_cat_array = enc.fit_transform(df)
            
            # Rebuild encoded columns
            # TODO: Make that less cryptic
            was_encoded = enc.transformers_[0][1].drop_idx_ == None
            i_encoded = np.where(was_encoded)[0]
            i_not_encoded = np.where(~was_encoded)[0]
            
            new_columns = []
            new_cat_features = []
            cat_encoded_features = []
            cat_not_encoded_features = []

            for feature in columns:
                # If the feature was among to-be-encoded candidates
                if feature in enc.transformers_[0][2]:
                    i = enc.transformers_[0][2].index(feature)
                    # If the feature actually was encoded
                    if i in i_encoded:
                        new_features = enc.transformers_[0][1].categories_[i]
                        # Avoid duplication by putting a prefix
                        new_features = [feature + "_" + str(f) for f in new_features]
                        new_columns.extend(list(new_features))
                        new_cat_features.extend(list(new_features))
                        cat_encoded_features.extend(list(new_features))
                    else:
                       new_columns.append(feature)
                       new_cat_features.append(feature)
                       cat_not_encoded_features.append(feature)
                else:
                    new_columns.append(feature)
            
            # Rebuild the dataframe
            df = pd.DataFrame(index=index, columns=new_columns)
            df[num_features] = df_num_array
            df[new_cat_features] = df_cat_array
                    
            # Reapply correct dtypes
            for f in new_columns:
                if f in num_features:
                    df[f] = df[f].astype(np.float64)
                elif f in new_cat_features:
                    df[f] = df[f].astype(np.int64)
        
            # Save the dataset
            df.to_csv(
                DATA_PATH + os.sep + "kdd_census_income_preprocessed.csv",
            )
        
        else:
            # Load and split the data
            file_path = DATA_PATH + os.sep + "kdd_census_income_preprocessed.csv"
            df = pd.read_csv(file_path, index_col=0)
            print(df)
        
        # Set the target and do some feature selection
        self.y = df.pop("income")
        self.X = df.drop([], axis=1)
        
        # Set default sensitive attributes
        self.sensitive_features = ["sex"]
        sensitive_values = [0]
        self.sensitive = sensitive_mask_from_features(
            self.X,
            self.sensitive_features,
            sensitive_values,
        )
    
    def objective(self, y_true, y_pred, sensitive):
        acc = accuracy_score(y_true, y_pred)
        p = p_rule(y_true, y_pred, sensitive)
       
        return max(acc, p)
