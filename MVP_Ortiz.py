# Libraries to evaluate our model's performance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklean.model_selection import KFold, cross_vaL_score

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

        Train : 0% to 80% of each users values.
        Validate : 80% to 90% of each users values.
        Test : 90% to 100% of each users values.
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

    '''
    # Set the defaults
    sns.set_context('talk')

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Plot the roc curve
    plot_roc_curve(clf, X, y, ax=ax1)

    # Plot the precision recall curve
    plot_precision_recall_curve(clf, X, y, ax=ax2)


def cross_validation(clf, X, y):
    '''
    This function accepts a classification model, X_set, y_set
    and returns cross validation metrics using ShuffleKFold

    This function is used to evaluate generalization performance of our models

    Parameters
    ----------
    classifier : Sklearn classification instance
    sklearn.ensemble, sklearn.linear_model, sklearn.tree, sklearn.neighbors, etc.
        A classification object that will be fit on X_train and y_train.

    X :  pandas DataFrame
        Accepts X_train dataset

    y : pandas DataFrame
        Accepts y_train dataset

    Returns
    -------
    
    
    '''
    # Create a KFold object to shuffle the data and set a random seed.
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)

    # Calculate cross-validation values for each k-splits
    cv_scores = cross_vaL_score(clf, X, y, cv=kfold)

    # Return the cv
    return cv_scores