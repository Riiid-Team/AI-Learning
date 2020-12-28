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
    This function accepts the training dataset
    Returns a graph with the percentage of questions, with and without explanations,
    answered correctly for parts 1-7 of the TOEIC test.
    '''
    
    part_explanation_acc = df.groupby(['part', 'question_had_explanation']).answered_correctly.mean()
    part_explanation_acc = pd.DataFrame(part_explanation_acc).reset_index()
    part_explanation_acc['part'] = part_explanation_acc.part.map({1: 'Photographs', 
                                                                  2: 'Question Response', 
                                                                  3: "Conversations", 
                                                                  4: "Talks(Narration)", 
                                                                  5: "Incomplete Sentences", 
                                                                  6: "Text Completion", 
                                                                  7: "Passages"})
    part_explanation_acc

    part_explanation_acc['sorting'] = [1,1,2,2,3,3,5,5,7,7,6,6,4,4]
    part_explanation_acc = part_explanation_acc.sort_values(by='sorting', ascending=False)

    sns.set_context('talk')

    plt.figure(figsize=(16, 9))
    sns.barplot(data=part_explanation_acc, x='answered_correctly', y='part', hue="question_had_explanation", 
                palette=['#d55e00', '#009e73'], ec='black')
    plt.xlabel("Percent of Correct Answers")
    plt.legend(bbox_to_anchor=(1, 1), title ='Had Explanation')
    plt.title("Students Perform Better with an Explanation",fontweight='bold', fontsize=23)
    plt.ylabel("")
    fmt = [f'{i:0.0%}' for i in np.linspace(0, 1, 11)] # Format you want the ticks, e.g. '40%'
    plt.xticks(np.linspace(0, 1, 11), labels=fmt)
    plt.show()

    
def user_lectures_graph(df):
    '''
    This function accepts the training dataset
    Returns a graph of sampled users with the number of lectures they've watched
    against their accuracy.
    '''
    fig = plt.figure(figsize=(14, 8))
    sample = df.sample(1000)
    x = sample.user_lectures_running_total
    y = sample.cum_accuracy
    plt.scatter(x, y, marker='o',color='#0080ff')
    plt.title("Students Benefit Little From Lectures", fontsize=20)
    plt.xlabel("Number of Lectures Students Watched", fontsize=15)
    plt.ylabel("Percent of Correct Answers", fontsize=15)
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(rotation=0, fontsize=15)
    plt.axhline(0.53, xmin=0, xmax=250, color='black', linestyle='-.', 
                label='Mean of Percent of Correct Answers')
    plt.axvline(4.1, ymin=0, ymax=1.0, color='black', linestyle='dotted', 
                label='Average of Lectures Students Watched')
    fmt = [f'{i:0.0%}' for i in np.linspace(0, 1, 11)] # Format you want the ticks, e.g. '40%'
    plt.yticks(np.linspace(0, 1, 11), labels=fmt)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--", label='Best Fit Line')

    plt.text(210,0.75, f'Slope = {p[1]:0.0}', fontsize=12)

    plt.legend(bbox_to_anchor=(1, 1), fontsize=12)
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

