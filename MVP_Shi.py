import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

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

def merge_with_stats(df_train, df_val_or_test):
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

    val_or_test = df_val_or_test.merge(content_stats[['mean_content_accuracy']], how='left', on='content_id')
    val_or_test = val_or_test.merge(task_stats[['mean_task_accuracy']], how='left', on='task_container_id')
    val_or_test = val_or_test.merge(timestamp_stats[['mean_timestamp_accuracy']], how="left", on="user_id")
    val_or_test = val_or_test.merge(priortime_stats[['mean_priortime_accuracy']], how="left", on="user_id")

    return train, val_or_test

def drop_columns(df):
    cols = ['user_id', 'row_id', 'timestamp', 'content_id', 'content_type_id', 'task_container_id', 
            'user_answer', 'prior_question_elapsed_time', 'prior_question_had_explanation']
    df.drop(columns=cols, inplace=True)
    return df

def fill_nulls(df):
    df.fillna(0.5, inplace=True)
    return df

def scale(train, test, columns_to_scale):
    
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    train.drop(columns=['mean_timestamp_accuracy', 'mean_priortime_accuracy', 'user_lectures_running_total', 'avg_user_q_time'], inplace=True)
    test.drop(columns=['mean_timestamp_accuracy', 'mean_priortime_accuracy', 'user_lectures_running_total', 'avg_user_q_time'], inplace=True)
    
    return scaler, train, test