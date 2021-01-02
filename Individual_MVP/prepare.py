# Import libraries
import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import Warnings Filter
import warnings
warnings.filterwarnings("ignore")

###################### Prepare Riiid Data ########################

def stats_content_id(df):
    '''
    Computes the students' historic content performance.
    '''
    content_stats = df.groupby('content_id').answered_correctly.agg(['mean',
                                                                     'count',
                                                                     'std',
                                                                     'median',
                                                                     'skew'])

    content_stats.columns = ['mean_content_accuracy',
                             'question_content_asked',
                             'std_content_accuracy', 
                             'median_content_accuracy',
                             'skew_content_accuracy']

    return content_stats.round(2)


def stats_task_container_id(df):
    '''
    Computes the students' historic task performance.
    '''
    task_stats = df.groupby('task_container_id').answered_correctly.agg(['mean',
                                                                         'count',
                                                                         'std',
                                                                         'median',
                                                                         'skew'])

    task_stats.columns = ['mean_task_accuracy',
                          'question_task_asked',
                          'std_task_accuracy', 
                          'median_task_accuracy',
                          'skew_task_accuracy']

    return task_stats.round(2)


def mean_tagcount_accuracy(df):
    '''
    Computes the mean accuracy according to questions with the same count of tags.
    '''
    tagcount_accuracy = df.groupby('tag_count').answered_correctly.mean().rename('mean_tagcount_accuracy').astype(np.float32)
    return tagcount_accuracy


def mean_tag_accuracy(df):
    '''
    Compute the mean accuracy according to the tag of a question
    '''
    tags_accuracy = df.groupby('tags').answered_correctly.mean().rename('mean_tags_accuracy').astype(np.float32)
    return tags_accuracy


def tag_features(train, validate, test):
    '''
    Accepts train, validate, test datasets and add to them the features mean tagcount accuracy and mean tag accuracy
    '''
    
    tagcount_accuracy = mean_tagcount_accuracy(train)
    tags_accuracy = mean_tag_accuracy(train)

    train = train.merge(tagcount_accuracy, how='left', on='tag_count')
    train = train.merge(tags_accuracy, how='left', on='tags')

    validate = validate.merge(tagcount_accuracy, how='left', on='tag_count')
    validate = validate.merge(tags_accuracy, how='left', on='tags')

    test = test.merge(tagcount_accuracy, how='left', on='tag_count')
    test = test.merge(tags_accuracy, how='left', on='tags')

    return train, validate, test


def merge_with_stats_train(df_train):
    '''
    Merges train, validate, and test with the new features generated from the train set.
    '''
    # Calculate historical content and task performance on the train set.
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    # Merge user statistics onto the train set.
    train = df_train.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    train = train.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return train


def merge_with_stats_valortest(df_train, df_val_or_test):
    '''
    To merger the train/validate/test with the new features generated from the train. 
    '''
    # Calculate historical content and task performance on the train set.
    content_stats = stats_content_id(df_train)
    task_stats = stats_task_container_id(df_train)

    # Merge user statistics onto the validation or test set.
    val_or_test = df_val_or_test.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    val_or_test = val_or_test.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')

    return val_or_test


def handle_null(df):
    '''
    This function fills the missing values 
    in the column prior_question_elapsed_time with False (boolean) 
    and fill the missing values in the column prior_question_elapsed_time
    with 0.
    '''
    # Fill nan values with False
    df.prior_question_had_explanation.fillna(False, inplace = True)
    
    # Fill nan values with 0.
    df.prior_question_elapsed_time.fillna(0, inplace = True)
    
    return df


def handle_inf(df):
    '''
    This function replaces np.inf values with 0.
    '''
    # Replace questions without an explanation from np.inf to 0.
    m = df.prior_question_elapsed_time.apply(lambda i: 0 if i == np.inf else i)
    df.prior_question_elapsed_time = m
    
    return df


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


def drop_columns(df):
    """
    Accepts df and drops various columns that are not needed for modeling.
    """
    cols = ['timestamp', 'user_id', 'content_id', 'task_container_id',
            'question_id', 'bundle_id', 'part', 'tags', 'tag_count']
    df = df.drop(columns=cols)
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
    Accepts train, validate, test and a list of columns to scale.
    Returns train, validate, and test with new scaled columns.
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
    
    # Drop original columns.
    train.drop(columns=columns_to_scale, inplace=True)
    validate.drop(columns=columns_to_scale, inplace=True)
    test.drop(columns=columns_to_scale, inplace=True)
    
    return train, validate, test


def boolean_to_num(df):
    """
    Accepts df. Converts True and False values into 1's and 0's resepectively, within the 
    question_had_explanation column.
    """
    df = df.fillna(False)
    m = df.question_had_explanation.apply(lambda i: 1 if i == True else 0)
    df.question_had_explanation = m
    return df


def part_bundle_features(train, validate, test):
    '''
    This function accepts the train, validate, and test sets.
    Returns new features on train, validate, and test using population stats from the training set.
    
    '''
    
    # Calculate the average accuracy for each unique bundle id
    bundle_accuracy = train.groupby(['bundle_id'])['answered_correctly'].mean().round(2).to_frame().reset_index()
    bundle_accuracy.columns = ['bundle_id', 'mean_bundle_accuracy']
    
    # Merge bundle mean accuracy as a feature to train, validate, and test
    merged_train = train.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    merged_validate = validate.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    merged_test = test.merge(bundle_accuracy, left_on='bundle_id', right_on='bundle_id', how='left')
    
    # Calculate the average part accuracy
    tag_accuracy = train.groupby(['part'])['answered_correctly'].agg(['mean']).round(2).reset_index()
    tag_accuracy.columns = ['part', 'mean_part_accuracy']
    
    # Merge average part accuracy
    train_df = merged_train.merge(tag_accuracy, left_on='part', right_on='part')
    validate_df = merged_validate.merge(tag_accuracy, left_on='part', right_on='part')
    test_df = merged_test.merge(tag_accuracy, left_on='part', right_on='part')

    return train_df, validate_df, test_df


def object_to_float(df):
    '''
    This function transforms object dtype columns to float32 dtype.
    '''
    # Select all columns with an object dtype.
    columns_to_transform = df.select_dtypes('O').columns
    
    # Cast object columns with an object dtype to float dtype
    for column in columns_to_transform:
        df[column] = df[column].astype(np.float32)
    
    return df


#################################### COMPLETE PREP FUNCTION ########################################

def prep_riiid(df_train, df_validate, df_test):
    """
    Accepts train, validate and test datasets and prepares the data for exploration and modeling.
    Returns unscaled and scaled versions of train, validate and test
    """
    
    # Drop the unused columns from questions.csv and lectures.csv
    cols = ['lecture_id',
            'tag',
            'lecture_part',
            'type_of',
            'question_id',
            'bundle_id',
            'correct_answer',
            'question_part',
            'tags']

    # Drop unused columns from train, validate, and test
    df_train = df_train.drop(columns = cols)
    df_validate = df_validate.drop(columns = cols)
    df_test = df_test.drop(columns = cols)
    
    # Add sam features
    train = sam_train_features(df_train)
    validate = sam_valtest_features(train, df_validate)
    test = sam_valtest_features(train, df_test)
    
    # Handle nulls
    train = handle_null(train)
    validate = handle_null(validate)
    test = handle_null(test)
    
    # Handle the inf values
    train = handle_inf(train)
    validate = handle_inf(validate)
    test = handle_inf(test)
    
    # Drop lecture rows
    train = drop_lecture_rows(train)
    validate = drop_lecture_rows(validate)
    test = drop_lecture_rows(test)

    # Reading questions csv
    df_ques = pd.read_csv('questions_with_tag_counts.csv', index_col=0)

    # Merge train/validate/test with df_ques
    train = train.merge(df_ques, how='left', left_on='content_id', right_on='question_id')
    validate = validate.merge(df_ques, how='left', left_on='content_id', right_on='question_id')
    test = test.merge(df_ques, how='left', left_on='content_id', right_on='question_id')

    # Drop the redundant columns to save memory
    train.drop(columns=['content_type_id', 'user_answer', 'prior_question_elapsed_time', 'correct_answer'], inplace=True)
    validate.drop(columns=['content_type_id', 'user_answer', 'prior_question_elapsed_time', 'correct_answer'], inplace=True)
    test.drop(columns=['content_type_id', 'user_answer', 'prior_question_elapsed_time', 'correct_answer'], inplace=True)
    
    # Add features: mean part accuracy and mean bundle accuracy
    train, validate, test = part_bundle_features(train, validate, test)
    
    # Add features: mean content accuracy and mean task accuracy
    train = merge_with_stats_train(train)
    validate = merge_with_stats_valortest(train, validate)
    test = merge_with_stats_valortest(train, test)
    
    # Add features: mean tagcount accuracy and mean tag accuracy    
    train, validate, test = tag_features(train, validate, test)
    
    # Fill nulls created from merging
    validate = fill_nulls(validate)
    test = fill_nulls(test)
    
    # Shift prior question had explanation to current question
    train.prior_question_had_explanation = train.prior_question_had_explanation.shift(-1)
    validate.prior_question_had_explanation = validate.prior_question_had_explanation.shift(-1)
    test.prior_question_had_explanation = test.prior_question_had_explanation.shift(-1)

    # Rename column
    train = train.rename(columns={"prior_question_had_explanation": "question_had_explanation"})
    validate = validate.rename(columns={"prior_question_had_explanation": "question_had_explanation"})
    test = test.rename(columns={"prior_question_had_explanation": "question_had_explanation"})

    # Drop the column q_time in train_s for modeling
    train_s = train.drop(columns='q_time')
    
    # Drop columns no longer needed for the purpose of modeling
    train_s = drop_columns(train_s)
    validate_s = drop_columns(validate)
    test_s = drop_columns(test)
    
    # Convert boolean to binary values 0/1
    train_s = boolean_to_num(train_s)
    validate_s = boolean_to_num(validate_s)
    test_s = boolean_to_num(test_s)
    
    # Convert numeric object to float dtype for modeling
    train_s = object_to_float(train_s)
    validate_s = object_to_float(validate_s)
    test_s = object_to_float(test_s)
    
    # scale columns
    columns_to_scale = ['user_lectures_running_total', 'avg_user_q_time']
    train_s, validate_s, test_s = scale(train_s, validate_s, test_s, columns_to_scale)
    
    # Return unscaled and scaled versions of train, validate, and test
    return train, validate, test, train_s, validate_s, test_s
    