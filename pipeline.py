import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Silence outdated numpy warning
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

'''
Define the functions and Pipeline Transformers that will encode the data preprocessing steps
required to train and predict using our preditive model. This enables repeatability of the
process and protects against sample leakage.
'''

def one_hot(input_df, columns):
    '''
    One-hot encode the provided list of columns and return a new copy of the data frame
    '''
    df = input_df.copy()

    for col in columns:
        dummies = pd.get_dummies(df[col].str.lower())
        dummies.drop(dummies.columns[-1], axis=1, inplace=True)
        df = df.drop(col, axis=1).merge(dummies, left_index=True, right_index=True)
    
    return df

def plot_roc(fpr, tpr, auc_score):
    plt.figure(figsize=(6, 6))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, 'b', label='AUC = {:5.2f}'.format(auc_score))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.01,1.01])
    plt.ylim([-0.01,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return

class FillTransformer(BaseEstimator, TransformerMixin):
    '''
    Impute NaN values

    # TODO: Parameterize so values can be imputed with -1, mean, median, or mode.
    '''
    def fit(self, X, y=None):
        self.fill_value = -1
        return self
    
    def transform(self, X):
        # paramaterize this with mean, median, mode, etc. 
        # fill with -1
        # TODO: make this fill dynamic for all columns?
        df = X.copy()
        
        df.fillna(self.fill_value, axis=1, inplace=True)

        return df

class OneHotTransformer(BaseEstimator, TransformerMixin):
    '''
    One-hot encode features
    '''

    def fit(self, X, y=None):
        df = one_hot(X, ['city', 'phone'])
        self.train_columns = df.columns
        
        return self
    
    def transform(self, X):
        df = X.copy()
        df = one_hot(df, ['city', 'phone'])

        # Remove untrained columns
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0
                
        # Add trained on columns
        for col in df.columns:
            if col not in self.train_columns:
                df.drop(col, axis=1, inplace=True)
        
        return df[self.train_columns]

class SetDataTypesTransformer(BaseEstimator, TransformerMixin):
    '''
    Set the correct data types for the features defined in this class.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Cast datetimes
        df['last_trip_dt'] = pd.to_datetime(df['last_trip_date'])
        df['signup_dt'] = pd.to_datetime(df['signup_date'])
        # NOTE: should we leave the old columns intact or remove/replace them?

        # Convert boolean to int
        df.loc[:, 'luxury_car_user'] = df['luxury_car_user'].map({False: 0, True: 1})

        return df

class GenPredictorTransformer(BaseEstimator, TransformerMixin):
    '''
    Accept the raw data frame of features and return the data frame with predictor columns added.

    Input Pandas DataFrame, outputs same.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Find max trip date
        df['today_dt'] = df['last_trip_dt'].max() # assuming someone used the service today.

        # Find inactive days
        df['inactive_timedelta'] = df['today_dt'] - df['last_trip_dt']
        df['inactive_days'] = df['inactive_timedelta'].dt.days  # number of inactive days

        # Create target classes
        df.loc[df['inactive_days'] > 30, 'churned'] = 1  
        df.loc[df['inactive_days'] <= 30, 'churned'] = 0

        return df

class DeriveFeaturesTransformer(BaseEstimator, TransformerMixin):
    '''
    Add derived features to DataFrame.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Identify users who have only used service during surge pricing;
        # possible strong indicator of churn
        df.loc[df['surge_pct'] >= 100.0, 'always_surge_users'] = 1
        df.loc[df['surge_pct'] < 100.0, 'always_surge_users'] = 0

        return df

class SelectFeaturesTransformer(BaseEstimator, TransformerMixin):
    '''
    Select features for model training
    '''
    def __init__(self):
        self.features = [
            'avg_dist',
            'avg_rating_by_driver',
            'avg_rating_of_driver',
            'avg_surge',
            'city',  # string
            'phone',  # string
            'surge_pct',
            'trips_in_first_30_days',
            'luxury_car_user',  # Bool
            'weekday_pct',
        ]

    # TODO: find out how to parameterize for grid search
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        return df[self.features]
