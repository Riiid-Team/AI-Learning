import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


def question_explanation_graph(df):
    '''
    This function accepts a training dataset and returns
    a plot of percentage questions of questions that are correct.
    
    Two Subgroups
    -------------
    Questions that had explanations: % answered correctly, % answered incorrectly
    Questions that did not have explanations: % answered correctly, % answered incorrectly    
    '''
    # Answered Correctly vs Prior Question Had Explanation
    prior_question = df.groupby(['question_had_explanation', 'answered_correctly']).agg({'answered_correctly': ['count']})
    
    questions_without_explanations = prior_question.iloc[:,0][:2]
    questions_with_explanations = prior_question.iloc[:,0][2:]
    
    # total number of questions with and without explanations
    total_questions_without_explanations = sum(questions_without_explanations)
    total_questions_with_explanations = sum(questions_with_explanations)

    # questions without explanations
    qwoe_incorrect = questions_without_explanations.iloc[0]/total_questions_without_explanations
    qwoe_correct = questions_without_explanations.iloc[1]/total_questions_without_explanations

    # questions with explanations
    qwe_incorrect = questions_with_explanations.iloc[0]/total_questions_with_explanations
    qwe_correct = questions_with_explanations.iloc[1]/total_questions_with_explanations

    df = pd.DataFrame({
        'Question_had_an_explanation': ['Incorrect', 'Correct'],
        'Explanation': [qwe_incorrect, qwe_correct],
        'No Explanation': [qwoe_incorrect, qwoe_correct]
    })
    tidy = df.melt(id_vars='Question_had_an_explanation')
    sns.set_context('talk')
    plt.figure(figsize=(13, 7))
    sns.barplot(x='variable', y='value', hue='Question_had_an_explanation', data=tidy)
    plt.xlabel('')
    plt.ylabel('%')
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0,1,11));


def rfe_ranker(train):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a logistic regression model.
    """
    non_target_vars = ['question_had_explanation', 'user_acc_mean','mean_content_accuracy', 'mean_task_accuracy',
    'user_lectures_running_total_scaled', 'avg_user_q_time_scaled']
    
    target_var = ['answered_correctly']
    
    # creating logistic regression object
    lr = LogisticRegression()

    # fitting logistic regression model to features 
    lr.fit(train[non_target_vars], train[target_var])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(lr, 1)

    # using rfe object to transform features 
    rfe.fit_transform(train[non_target_vars], train[target_var])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_train = train[non_target_vars]

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

