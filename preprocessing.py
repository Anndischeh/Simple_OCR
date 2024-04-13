import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class Reshaper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_reshaped = X.reshape(-1, 28, 28, 1)
        return X_reshaped

def preprocessing(ds_train, ds_test, x_train, y_train, x_test, y_test):
    # Define column names
    num_columns = 785
    columns = ['label'] + list(range(1, num_columns))

    # Assign column names to the DataFrames
    ds_train.columns = columns
    ds_test.columns = columns

    # Reset index
    ds_train = ds_train.reset_index(drop=True)
    ds_test = ds_test.reset_index(drop=True)

    # Extract features and labels
    x_train_processed = x_train
    y_train_processed = np.eye(62)[y_train.astype(int)]
    x_test_processed = x_test
    y_test_processed = np.eye(62)[y_test.astype(int)]

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: StandardScaler
        ('reshaper', Reshaper())  # Step 2: Custom Reshaper
    ])

    x_train_processed = pipeline.fit_transform(x_train_processed)
    x_test_processed = pipeline.transform(x_test_processed)

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed
