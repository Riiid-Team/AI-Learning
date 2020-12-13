import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression

def rfe_ranker(train):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a logistic regression model.
    """
    non_target_vars = ['prior_question_had_explanation', 'user_acc_mean',
       'mean_content_accuracy', 'mean_task_accuracy',
       'mean_timestamp_accuracy_scaled', 'mean_priortime_accuracy_scaled',
       'user_lectures_running_total_scaled', 'avg_user_q_time_scaled']
    
    target_var = ['answered_correctly']
    
    # creating logistic regression object
    lr = LogisticRegression()

    # fitting logistic regression model to features 
    lr.fit(train[non_target_vars], train[target_var])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(lr, 1)

    # using rfe object to transform features 
    x_rfe = rfe.fit_transform(train[non_target_vars], train[target_var])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_train = train[non_target_vars]

    # creating list of the top features per rfe
    rfe_features = rfe_train.loc[:,feature_mask].columns.tolist()

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = rfe_train.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # return df sorted by rank
    return rfe_ranks_df.sort_values('Rank')


def KBest_ranker(X, y, n):
   """
   Returns the top n selected features based on the SelectKBest calss
   Parameters: predictors(X) in df, target(y) in df, the number of features to select(n)
   """
   f_selector = SelectKBest(f_regression, k=n)
   f_selector = f_selector.fit(X, y)
   f_support = f_selector.get_support()
   f_feature = X.iloc[:, f_support].columns.tolist()
   return f_feature

def feature_over_time(feature, train, agg_method):
    feature_on_time = pd.DataFrame(train.groupby("timestamp")[feature].agg([agg_method]))
    feature_on_time.reset_index(inplace=True)
    feature_on_time.rename(columns={agg_method:feature}, inplace=True)
    
    feature_on_time['seconds'] = (feature_on_time.timestamp/1000).round(0)
    feature_on_time['minutes'] = (feature_on_time.timestamp/(1000*60)).round(0)
    feature_on_time['hours'] = (feature_on_time.timestamp/(1000*60*60)).round(0)
    feature_on_time['days'] = (feature_on_time.timestamp/(1000*60*60*24)).round(0)
    feature_on_time['months'] = (feature_on_time.timestamp/(1000*60*60*24*30)).round(0)
    feature_on_time['years'] = (feature_on_time.timestamp/(1000*60*60*24*30*12)).round(0)
    
    return feature_on_time