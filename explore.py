import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif


################################# Riiid Visualizations ###########################################
def question_explanation_graph(df):
    '''
    This function accepts a training dataset and returns
    a plot of the percentage questions answered correctly v. incorrectly.
    
    Two Subgroups
    -------------
    Questions that have explanations: % answered correctly, % answered incorrectly
    Questions that do not have explanations: % answered correctly, % answered incorrectly    
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

    # Create a dataframe of percentages.
    df = pd.DataFrame({
        'Question_had_an_explanation': ['Incorrect', 'Correct'],
        'Explanation': [qwe_incorrect, qwe_correct],
        'No Explanation': [qwoe_incorrect, qwoe_correct]
    })
    
    # Melt the column
    tidy = df.melt(id_vars='Question_had_an_explanation')
    
    # Plotting
    sns.set_context('talk')
    plt.figure(figsize=(13, 7))
    sns.barplot(x='variable', y='value', hue='Question_had_an_explanation', data=tidy, palette=['#d55e00', '#009e73'], ec='black')
    plt.title("Students Perform Better On Questions With Explanations",fontsize=20) 
    plt.legend() 
    plt.xlabel('')
    plt.ylabel('%',fontsize=15)
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0,1,5))
    plt.show()

    
###################################### Feature Selection ############################################
def rfe_ranker(X_train, y_train):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a logistic regression model.
    """
    
    # Create a Logistic Regression object
    lr = LogisticRegression()

    # Fit the object with the training set
    lr.fit(X_train, y_train)

    # Create a Recursive Feature Elimination object. Rank 1 feature as the most important.
    rfe = RFE(lr, 1)

    # Fit the RFE object with the training data to 
    rfe.fit_transform(X_train, y_train)

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = X_train.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'feature': feature_names,
                                 'rank': feature_ranks})

    # return df sorted by rank
    rfe_ranked_featuers = rfe_ranks_df.sort_values('rank').reset_index(drop=True)
    
    return rfe_ranked_featuers.head()



def KBest_ranker(X, y, n):
    '''
    Returns the top n selected features with their scores based on the SelectKBest calss
    Parameters: scaled predictors(X) in df, target(y) in df, the number of features to select(n)
    '''

    # parameters: f_regression stats test, give me 5 features
    f_selector = SelectKBest(f_classif, k=n)

    # Fit on X and y
    f_selector.fit(X, y)

    # boolean mask of whether the column was selected or not. 
    feature_score = f_selector.scores_.round(2)

    # Put the features in a dataframe
    df_features = pd.DataFrame({'features': X.columns, 
                                'score': feature_score})

    # Sort the features based on their score
    df_features.sort_values(by="score", ascending=False, inplace=True, ignore_index=True)

    # Compute how many features in X
    m = X.shape[1]
    
    # Add a rank column
    df_features['rank'] = range(1, m+1)
    
    return df_features[:n]


def feature_over_time(feature, train, agg_method):
    '''
    Accepts 
    '''
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

