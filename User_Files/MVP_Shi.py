import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_roc_curve, roc_auc_score

def sample_train(df, n):
    '''
    Accept the train.csv and an positive integer n
    and return the records of the first n users
    '''
    user_ids = df.user_id.value_counts().sort_index().iloc[0: n]
    user_ids = user_ids.index.to_list()
    df = df.set_index("user_id")
    sample = df.loc[user_ids]
    return sample

def handle_null(df):
    '''
    This function is going to fill the missing values 
    in the column prior_question_elapsed_time with False (boolean) 
    and fill the missing values in the column prior_question_elapsed_time
    with 0.
    '''
    df.prior_question_had_explanation.fillna(False, inplace = True)
    df.prior_question_elapsed_time.fillna(0, inplace = True)
    return df

def handle_inf(df):
    m = df.prior_question_elapsed_time.apply(lambda i: 0 if i == np.inf else i)
    df.prior_question_elapsed_time = m
    return df

def drop_lecture_rows(df):
    '''
    Drop the lecture rows in the dataframe
    '''
    mask = df['answered_correctly'] != -1
    df = df[mask]
    return df
    
def sample_split(df):
    '''
    This function is used to quickly split the sample from train.csv
    to train and test at ratio of 8: 2
    If the users who have less than 10 records, the function will print out the user ids
    and drop their records.
    '''
    train = pd.DataFrame()
    test = pd.DataFrame()
    train_size = 0.8
    user_ids = df.index.value_counts().sort_index().index
    for user_id in user_ids:
        if df.loc[[user_id]].shape[0] <10:
            continue
        elif df.loc[[user_id]].shape[0] >=10: 
            df_user = df.loc[[user_id]]
            n = df_user.shape[0]
            test_start_index = round(train_size * n)
            df_train = df_user.iloc[:test_start_index]
            df_test = df_user.iloc[test_start_index:]     
            train = pd.concat([train, df_train])
            test = pd.concat([test, df_test])
    
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    return train, test

def stats_content_id(df):
    '''
    To compute the students' historic performance regarding the content
    '''
    content_stats = df.groupby('content_id').answered_correctly.agg(['mean', 'count', 'std', 'median', 'skew'])

    content_stats.columns = ['mean_content_accuracy', 'question_content_asked', 'std_content_accuracy', 
                                'median_content_accuracy', 'skew_content_accuracy']

    return content_stats

def stats_task_container_id(df):
    '''
    To compute the students' historic performance on tasks
    '''
    task_stats = df.groupby('task_container_id').answered_correctly.agg(['mean', 'count', 'std', 'median', 'skew'])

    task_stats.columns = ['mean_task_accuracy', 'question_task_asked', 'std_task_accuracy', 
                            'median_task_accuracy', 'skew_task_accuracy']

    return task_stats

def stats_timestamp(df):
    '''
    Compute the average time that take the user to answer one question. 
    '''
    timestamp_stats = df.groupby("user_id").timestamp.agg(['mean', 'count', 'std', 'median', 'skew'])
    timestamp_stats.columns = ['mean_timestamp_accuracy', 'question_timestamp_asked', 'std_timestamp_accuracy', 
                                'median_timestamp_accuracy', 'skew_timestamp_accuracy']

    return timestamp_stats

def stats_priortime(df):
    '''
    Compute the avereage of the column prior_question_elapsed_time
    '''
    priortime_stats = df.groupby("user_id").prior_question_elapsed_time.agg(['mean', 'count', 'std', 'median', 'skew'])
    priortime_stats.columns = ['mean_priortime_accuracy', 'question_priortime_asked', 'std_priortime_accuracy', 
                                'median_priortime_accuracy', 'skew_priortime_accuracy']

    return priortime_stats


def merge_with_stats_train(df_train):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)
    timestamp_stats = stats_timestamp(df_train)
    priortime_stats = stats_priortime(df_train)

    train = df_train.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    train = train.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')
    train = train.merge(timestamp_stats[['mean_timestamp_accuracy']], how="left", on="user_id")
    train = train.merge(priortime_stats[['mean_priortime_accuracy']], how="left", on="user_id")

    return train


def merge_with_stats_valortest(df_train, df_val_or_test):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)
    timestamp_stats = stats_timestamp(df_train)
    priortime_stats = stats_priortime(df_train)

    val_or_test = df_val_or_test.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    val_or_test = val_or_test.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')
    val_or_test = val_or_test.merge(timestamp_stats[['mean_timestamp_accuracy']], how="left", on="user_id")
    val_or_test = val_or_test.merge(priortime_stats[['mean_priortime_accuracy']], how="left", on="user_id")

    return val_or_test

def drop_columns_train(df):
    cols = ['user_id', 'timestamp', 'content_id', 'content_type_id', 
            'task_container_id', 'user_answer', 'last_q_time', 'prior_question_elapsed_time']
    df.drop(columns=cols, inplace=True)
    return df

def drop_columns_valortest(df):
    cols = ['user_id', 'timestamp', 'content_id', 'content_type_id', 
            'task_container_id', 'user_answer', 'prior_question_elapsed_time']
    df.drop(columns=cols, inplace=True)
    return df

def fill_nulls(df):
    df.fillna(0.5, inplace=True)
    return df

def scale(train, validate, test, columns_to_scale):
    
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)

    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)

    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    train.drop(columns=['mean_timestamp_accuracy', 'mean_priortime_accuracy', 'user_lectures_running_total', 'avg_user_q_time'], inplace=True)
    validate.drop(columns=['mean_timestamp_accuracy', 'mean_priortime_accuracy', 'user_lectures_running_total', 'avg_user_q_time'], inplace=True)
    test.drop(columns=['mean_timestamp_accuracy', 'mean_priortime_accuracy', 'user_lectures_running_total', 'avg_user_q_time'], inplace=True)
    
    return scaler, train, validate, test

def boolean_to_num(df):
    m = df.prior_question_had_explanation.apply(lambda i: 1 if i == True else 0)
    df.prior_question_had_explanation = m
    return df

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
    clf: the classification algorithm after fitting with X_train, y_train
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
    
    metrics = pd.DataFrame()

    for name, clf in zip(names, classifiers):
        
        # Set up a progress indicator        
        print(f"Currently runnig on model {name}")
        
        # Working on thr train dataset        
        clf = clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_train)
        y_proba, score = auc_score_proba(clf, X_train, y_train)
        d = {"Algo": name, "dataset": "train", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)
        
        # Working on the validate dataset
        y_proba, score = auc_score_proba(clf, X_validate, y_validate)
        d = {"Algo": name, "dataset": "validate", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)
             
        # Working on the test dataset
        y_proba, score = auc_score_proba(clf, X_test, y_test)
        d = {"Algo": name, "dataset": "test", "AUC score": score}
        metrics = metrics.append(d, ignore_index=True)
             
    return metrics