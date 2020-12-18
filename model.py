import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score


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
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Plot the precision recall curve
    plot_precision_recall_curve(clf, X, y, ax=ax2)

    
def auc_curve_plot1(clf, algo_name, X, y):
    '''
    This function accepts y_validate or y_test and predict_proba or decision_function classifier attribute
    Example:
    ---------
    auc_mertic(actual_outcome=y_validate, decision_func=svc.decision_function(X_validate))
    Returns a plot of the ROC curve.
    Parameters
    ----------
    clf : The classification model fit with X_train, y_train
    X : X_validate or X_test
    y : y_validate or y_test
    Returns
    -------
    A visualization of the roc for a given classification model.
    '''
    # Set visualization defaults
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the roc curve
    plot_roc_curve(clf, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set_title(f"ROC curve for {algo_name}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    
def auc_score_proba(clf, X, y):
    '''
    This function accepts a classification model that can estimate probability, X_set and y_set
    and returns a dataframe of predicted probabilty on a binary class
    and also returns a auc score
    
    Parameter
    ----------
    clf: the classification algorithm after fitting on X_train, y_train
    X: X_train, X_validate and X_test
    y: y_train, y_validate and y_test
    
    Returns
    ----------
    1. A dataframe containing the probability estimates
    2. AUC score
    '''
    y_proba = clf.predict_proba(X)
    y_proba = pd.DataFrame(y_proba, columns=['p_0', 'p_1'])
    score = roc_auc_score(y, y_proba['p_1'])
    return y_proba, score


def model_multiple_algos(names, classifiers, X_train, y_train, X_validate, y_validate, X_test, y_test):
    '''
    This function accetps a list of classifiers, feature dataset and target dataset
    and return the auc scores.
    The order of the names should match the order of the classifiers

    Parameter
    ----------
    names: a list of the names of the classifiers that will be tested.
    classifiers: a list of classifier objects.
    X_train: features in the train dataset
    y_train: target variable in the train dataset
    X_validate: features in the validate dataset
    y_validate: target variable in the validate dataset
    X_test: features in the test dataset
    y_test: target variable in the test dataset

    Example
    ----------
    names: ["logistic Regression", "Decision Tree"]
    classifiers: [LogisticRegression(), DecisionTreeClassifier(max_depth=3)]
    all the datasets ready for modeling

    Return
    ----------
    A dataframe of auc scores associated with the classification algorithm and the dataset it used. 
    '''
    
    metrics = pd.DataFrame()

    for name, clf in zip(names, classifiers):
        
        # Set up a progress indicator        
        print(f"Currently running on model {name}") 
        
        # Fit on the train dataset        
        clf = clf.fit(X_train, y_train)
        
        # Compute the AUC score on train
        y_proba, score = auc_score_proba(clf, X_train, y_train)
        d = {"Algo": name, "dataset": "train", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)
        
        # Compute the AUC score on validate
        y_proba, score = auc_score_proba(clf, X_validate, y_validate)
        d = {"Algo": name, "dataset": "validate", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)
             
        # Compute the AUC score on test
        y_proba, score = auc_score_proba(clf, X_test, y_test)
        d = {"Algo": name, "dataset": "test", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)

        # Show the completeness of the modeling
        print(f"{name} has completed")
             
    return metrics