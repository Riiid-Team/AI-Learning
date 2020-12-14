# Libraries to evaluate our model's performance
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import KFold, cross_val_score


######################### prepare.py functions #################################

def split_users(df, sample=True, train_size=.8, validate_size=.1):
    '''
    This function accepts data from `acquire.get_riiid_data()`
    and returns lists of train, validate and test user ids.

    Parameters
    ----------
    df : pandas.core.DataFrame
        df accepts data from acquire.get_riiid_data()

    sample : boolean optional, default True
        If sample is True, then return a sample of users
    
    train_size : float optional, default .8
        The percentage of the dataset assigned to train
    
    validate_size : float optional, default .1
        The percentage of the dataset assigned to validate

    Returns
    -------
    train_ids, validate_ids, test_ids : list
        A list of users ids assigned to train, validate, and test
    '''
    # Set a random seed to reproduce splits
    random.seed(123)
    
    if sample == True:
        # Gather a random sample of 100_000 user ids
        user_ids = list(random.sample(df['user_id'].unique(), 100_000))
    else:
        # Gather all user ids
        user_ids = list(df['user_id'].unique())
    
    # Calculate the number of users
    total_num = len(user_ids)
    
    # Calculate the number of users in train, validate. The remaining will go in test
    train_num = int(total_num*train_size)
    validate_num = math.ceil(total_num*validate_size)
    
    # Randomly select 80% of the users to be in train.
    train_ids = random.sample(user_ids, train_num)
    
    # Remove user_ids assigned to the training set.
    remaining_val_test_users = list(set(user_ids) - set(train_ids))
    
    # Assign the remaining user ids to validate and test.
    validate_ids = random.sample(remaining_val_test_users, validate_num)
    test_ids = list(set(remaining_val_test_users) - set(validate_ids))
    
    # Return the users assigned to train, validate, and test
    return train_ids, validate_ids, test_ids


def train_validate_test(df, sampled=True, train_pct=.8, validate_pct=.1):
    '''
    This function accepts data from `acquire.get_riiid_data()`
    and returns train, validate, and test sets.

    Parameters
    ----------
    df : pandas.core.DataFrame
        df accepts data from acquire.get_riiid_data()

    sampled : boolean optional, default True
        If sampled is True, then only return a sample of users
    
    train_size : float optional, default .8
        The percentage of the dataset assigned to train
    
    validate_size : float optional, default .1
        The percentage of the dataset assigned to validate

    Returns
    -------
    train, validate, test

    '''
    # Randomly seperate users into train, validate, and test
    train_ids, validate_ids, test_ids = split_users(df, sample=sampled, train_size=train_pct, validate_size=validate_pct)

    # Split the dataset using the assigned user ids
    train = df.loc[df['user_id'].isin(train_ids)]
    validate = df.loc[df['user_id'].isin(validate_ids)]
    test = df.loc[df['user_id'].isin(test_ids)]

    return train, validate, test


######################### model.py functions #################################

def auc_curve_plot(clf, X, y):
    '''
    This function accepts y_validate or y_test and predict_proba or decision_function classifier attribute
    Example:
    ---------
    auc_mertic(actual_outcome=y_validate, decision_func=svc.decision_function(X_validate))

    Returns a plot of the ROC curve and Precision Recall Curve.

    Parameters
    ----------
    clf : The classification model fit with X_train, y_train

    X : X_validate or X_test

    y : y_validate or y_test

    Returns
    -------
    A visualization of the roc and precision recall curves for a given classification model.
    '''
    # Set visualization defaults
    sns.set_context('talk')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Plot the roc curve
    plot_roc_curve(clf, X, y, ax=ax1)

    # Plot the precision recall curve
    plot_precision_recall_curve(clf, X, y, ax=ax2)


def cross_validation(clf, X, y, n_folds=5):
    '''
    This function accepts a classification model, X_set, y_set
    and returns cross validation metrics using ShuffleKFold

    This function is used to evaluate generalization performance of our models

    Parameters
    ----------
    classifier : Sklearn classifier instance
    sklearn.ensemble, sklearn.linear_model, sklearn.tree, sklearn.neighbors, etc.
        A classification object that will be fit on X_train and y_train.

    X :  pandas DataFrame
        Accepts X_train dataset

    y : pandas DataFrame
        Accepts y_train dataset

    n_folds : integer optional, default 5
         The number of models, folds, and splits of the training data.

    Returns
    -------
    cv_scores : pandas.core.DataFrame
        Cross validation scores for n number of folds.
    
    '''
    # Create a KFold object to shuffle the data and set a random seed.
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=123)

    # Calculate cross-validation values for each k-splits
    cv_scores = cross_val_score(clf, X, y, cv=kfold)

    # Return the cv
    return cv_scores

########################## DO NOT USE ###############################################
def data_split(df):
    '''
    This function accepts the prepared Riiid dataset and
    returns train, validate, and test sets.

    ATTENTION USER:
    This function will split 100_000 users data in train, validate, and test.


    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        CREATE FUNCTION
        Dataframe from prepare.prep_riiid_data()

    Returns
    -------
    train, validate, test : pandas.core.frame.DataFrame
    (Data is split on an index level per user)

        Train : 0% to 80% of each users sequential values.
        Validate : 80% to 90% of each users sequential values.
        Test : 90% to 100% of each users sequential values.
    '''
    # Create seperate dataframes to store train, validate, and test data
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    # Set up train and validate size
    train_size = 0.8
    validate_size = 0.1

    # Store user ids in a list
    sampled_ids = df.user_id.unique()
    
    # For each user, split their data into train, validate, and test.
    for user in sampled_ids:
        # Locate the user
        data = df.loc[df['user_id'] == user]

        # Calculate the number of observations they have.
        n = data.shape[0]

        # Use a percentage based method to split using index.
        # 80-10-10 split
        train_end_index = int(train_size * n)
        validate_end_index = train_end_index + int(validate_size * n)

        # Locate the indexes for train, validate, and test.
        df_train = data.iloc[:train_end_index]
        df_validate = data.iloc[train_end_index:validate_end_index]
        df_test = data.iloc[validate_end_index:]

        # Concatenate a users split data into train, validate, and test.
        train = pd.concat([train, df_train])
        validate = pd.concat([validate, df_validate])
        test = pd.concat([test, df_test])

    # Cache datasets for quicker iteration
    train.to_csv('train_explore.csv', index=False)
    validate.to_csv('validate.csv', index=False)
    test.to_csv('test.csv', index=False)
        
    return train, validate, test
