# Prepare File 
# Imports

import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Warnings 
import warnings
warnings.filterwarnings("ignore")

###################### Prepare Riiid Data ########################

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

def merge_with_stats_train(df_train):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    train = df_train.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    train = train.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return train


def merge_with_stats_valortest(df_train, df_val_or_test):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    val_or_test = df_val_or_test.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    val_or_test = val_or_test.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return val_or_test

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
    
def merge_with_stats_train(df_train):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    train = df_train.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    train = train.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return train

def merge_with_stats_valortest(df_train, df_val_or_test):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''   
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    val_or_test = df_val_or_test.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    val_or_test = val_or_test.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return val_or_test

def sam_train_features(df):
    """
    Accepts train df. Prepares data with several changes outlined in notebook
    """
    
    # average user accuracy
    user_acc_mean = pd.DataFrame(df.groupby('user_id')['answered_correctly'].mean())
    user_acc_mean.columns = ['user_acc_mean']
    df = df.merge(user_acc_mean, how = 'left', left_on = 'user_id', right_on = 'user_id')
    
    # running count of lectures viewed per user
    df['user_lectures_running_total'] = df.groupby(by=['user_id'])['content_type_id'].transform(lambda x: x.cumsum())
    
    # time taken to answer previous question
    lastq = pd.DataFrame(df.timestamp.diff())

    lastq.columns = ['q_time']

    lastq.fillna(0)

    df = pd.concat([df, lastq], axis=1)
    
    df.q_time = df.q_time.shift(-1)

    df['q_time'] = np.where((df.q_time.isnull()), 0, df.q_time)
    df['q_time'] = np.where((df.q_time < 0), 0, df.q_time)
    
    for x in range(0,10):
        # set to loop number of times == largest count of questions in bundle
        df['q_time'] = np.where((df.q_time ==  0) & (df.prior_question_elapsed_time >= 0), df.q_time.shift(1), df.q_time)
    
    # avg time each user takes a question
    avg_q_time_user = pd.DataFrame(df.groupby('user_id').mean().round()['q_time'])
    
    avg_q_time_user.columns = ['avg_user_q_time']
    
    df = df.merge(avg_q_time_user, how='left', left_on='user_id', right_on='user_id')

    return df

def sam_valtest_features(df, val_or_test):
    """
    Accepts train df along with validate or test df. Prepares data with several changes outlined in notebook
    """
    
    # average user accuracy
    user_acc_mean = pd.DataFrame(df.groupby('user_id')['answered_correctly'].mean())
    user_acc_mean.columns = ['user_acc_mean']
    val_or_test = val_or_test.merge(user_acc_mean, how = 'left', left_on = 'user_id', right_on = 'user_id')
    
    # running count of lectures viewed per user
    val_or_test['user_lectures_running_total'] = val_or_test.groupby(by=['user_id'])['content_type_id'].transform(lambda x: x.cumsum())
    
    # avg time each user takes a question
    avg_q_time_user = pd.DataFrame(df.groupby('user_id').mean().round()['q_time'])
    
    # renaming column
    avg_q_time_user.columns = ['avg_user_q_time']
    
    # merging data
    val_or_test = val_or_test.merge(avg_q_time_user, how='left', left_on='user_id', right_on='user_id')
    
    # returning df
    return val_or_test

def drop_columns_train(df):
    """
    Accepts df. Drops various columns from train that are not needed after being used for other purposes.
    """
    cols = ['user_id', 'content_id', 'content_type_id', 'timestamp',
            'task_container_id', 'user_answer', 'prior_question_elapsed_time']
    df.drop(columns=cols, inplace=True)
    return df

def drop_columns_valortest(df):
    """
    Accepts df. Drops various columns from validate df or test df that are not needed after being used for other purposes.
    """
    cols = ['user_id', 'timestamp', 'content_id', 'content_type_id', 
            'task_container_id', 'user_answer', 'prior_question_elapsed_time']
    df.drop(columns=cols, inplace=True)
    return df

def drop_lecture_rows(df):
    '''
    Drop the lecture rows from the dataframe.
    '''
    mask = df['answered_correctly'] != -1
    df = df[mask]
    return df

def fill_nulls(df):
    '''
    Fills nulls with .5
    '''
    df.fillna(0.5, inplace=True)
    return df

def scale(train, validate, test, columns_to_scale):
    '''
    Accepts train, validate, test and list of columns to scale. Scales listed columns.
    '''
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
    
    train.drop(columns=columns_to_scale, inplace=True)
    validate.drop(columns=columns_to_scale, inplace=True)
    test.drop(columns=columns_to_scale, inplace=True)
    
    return scaler, train, validate, test

def boolean_to_num(df):
    """
    Accepts df. Converts True and False values into 1's and 0's resepectively, within the 
    question_had_explanation column.
    """
    m = df.question_had_explanation.apply(lambda i: 1 if i == True else 0)
    df.question_had_explanation = m
    return df


def tag_bundle_features(train, validate, test):
    '''
    This function accepts the train, validate, and test sets.
    Returns new features on train, validate, and test using population stats from the training set.
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    '''
    
    # Calculate the average accuracy for each unique bundle id
    bundle_accuracy = train.groupby(['bundle_id'])['answered_correctly'].mean().round(2).to_frame().reset_index()
    bundle_accuracy.columns = ['bundle_id', 'mean_bundle_accuracy']
    
    # Add bundle mean accuracy as a feature to train, validate, and test
    merged_train = train.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    merged_validate = validate.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    merged_test = test.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    
    # Calculate the average part accuracy
    tag_accuracy = train.groupby(['question_part'])['answered_correctly'].agg(['mean']).round(2).reset_index()
    tag_accuracy.columns = ['question_part', 'mean_part_accuracy']
    
    # Add average part accuracy
    train_df = merged_train.merge(tag_accuracy, left_on='question_part', right_on='question_part')
    validate_df = merged_validate.merge(tag_accuracy, left_on='question_part', right_on='question_part')
    test_df = merged_test.merge(tag_accuracy, left_on='question_part', right_on='question_part')
    
    # Calculate the mean container accuracy for each part
    tag_bundles = train.groupby(['question_id', 'task_container_id', 'question_part'])['answered_correctly'].mean().round(2).reset_index()
    tag_bundles.rename(columns={'answered_correctly': 'mean_container_part_accuracy'}, inplace=True)
    
    # Add mean container part accuracy
    train_set = train_df.merge(tag_bundles, how='left', 
                               left_on=['task_container_id', 'question_part'], 
                               right_on=['task_container_id', 'question_part'])
    
    validate_set = validate_df.merge(tag_bundles, how='left', 
                                     left_on=['task_container_id', 'question_part'], 
                                     right_on=['task_container_id', 'question_part'])
    
    test_set = test_df.merge(tag_bundles, how='left', 
                             left_on=['task_container_id', 'question_part'], 
                             right_on=['task_container_id', 'question_part'])

    
    return train_set, validate_set, test_set

####### COMPLETE PREP FUNCTION ########

def prep_riiid(df_train, df_validate, df_test):
    """
    Accepts train, validate and test DFs. Returns all three fully prepped for exploration.
    """
    # Add new features to train, validate, and test
    df_train, df_validate, df_test = tag_bundle_features(df_train, df_validate, df_test)
    
    # Drop the columns merged from questions.csv and lectures.csv
    cols = ['lecture_id', 'tag', 'lecture_part', 'type_of', 'question_id',
            'bundle_id', 'correct_answer', 'question_part', 'tags']

    df_train = df_train.drop(columns = cols)
    df_validate = df_validate.drop(columns = cols)
    df_test = df_test.drop(columns = cols)
    
    # add sam features
    train = sam_train_features(df_train)
    validate = sam_valtest_features(train, df_validate)
    test = sam_valtest_features(train, df_test)
    
    # handle nulls
    train = handle_null(train)
    validate = handle_null(validate)
    test = handle_null(test)
    
    # Handle the inf values
    train = handle_inf(train)
    validate = handle_inf(validate)
    test = handle_inf(test)
    
    # drop lecture rows
    train = drop_lecture_rows(train)
    validate = drop_lecture_rows(validate)
    test = drop_lecture_rows(test)
    
    # Merge the new features genereated from Shi
    train = merge_with_stats_train(train)
    validate = merge_with_stats_valortest(train, validate)
    test = merge_with_stats_valortest(train, test)
    
    # drop columns no longer needed
    train = drop_columns_train(train)
    validate = drop_columns_valortest(validate)
    test = drop_columns_valortest(test)
    
    # fill nulls created from merging
    validate = fill_nulls(validate)
    test = fill_nulls(test)

    # shift prior question had explanation to current question
    train.prior_question_had_explanation = train.prior_question_had_explanation.shift(-1)
    validate.prior_question_had_explanation = validate.prior_question_had_explanation.shift(-1)
    test.prior_question_had_explanation = test.prior_question_had_explanation.shift(-1)

    train = train.rename(columns={"prior_question_had_explanation": "question_had_explanation"})
    validate = validate.rename(columns={"prior_question_had_explanation": "question_had_explanation"})
    test = test.rename(columns={"prior_question_had_explanation": "question_had_explanation"})

    # convert boolean to num
    train = boolean_to_num(train)
    validate = boolean_to_num(validate)
    test = boolean_to_num(test)
    
    # scale columns
    columns_to_scale = ['user_lectures_running_total', 'avg_user_q_time']

    scaler, train_s, validate_s, test_s = scale(train, validate, test, columns_to_scale)

    # drop the column q_time in train_s for modeling
    train_s.drop(columns='q_time', inplace=True)
    
    # returning DFs
    return train, validate, test, train_s, validate_s, test_s
    