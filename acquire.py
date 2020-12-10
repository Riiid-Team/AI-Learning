# Acquire File
# Imports 

import pandas as pd
import numpy as np
import random
import os

random.seed(123)

###################### Acquire Riiid Data ########################
def get_riiid_data():
    '''
    This function acquires and merges `train`, `lectures`, and `questions` datasets and
    returns a dataframe.
    
    The merged dataset only contains HALF of users.

    Parameters
    ----------
    None 
    
    Returns
    -------
    df_filtered : pandas.core.frame.DataFrame
    '''
    if os.path.isfile('riiid_data.csv'):
        return pd.read_csv('riiid_data.csv', index_col=False)
    else:
        # Dictionaries to cast column dtypes.
        _, lecture_dtypes, question_dtypes = datatype_converter()
        
        # Acquire data
        df_train = sampled_train()
        df_lectures = pd.read_csv('lectures.csv', dtype=lecture_dtypes)
        df_questions = pd.read_csv('questions.csv', dtype=question_dtypes)
        
        # Left join df_train and df_lectures using `content_id` as the primary key.
        df_merged = df_train.merge(df_lectures, left_on='content_id', right_on='lecture_id', how='left')
        
        # Left join df_merged and df_questions using `content_id` as the primary key.
        df = df_merged.merge(df_questions, left_on='content_id', right_on='question_id', how='left')
    
        # Fill missing values with 0.
        df_filtered = df.fillna(0)
        
        # Change the data types of numeric columns.
        df_filtered.lecture_id = df_filtered.lecture_id.astype(np.int16)
        df_filtered.tag = df_filtered.tag.astype(np.int8)
        df_filtered.part_x = df_filtered.part_x.astype(np.int8)
        df_filtered.part_y = df_filtered.part_y.astype(np.int8)
        df_filtered.question_id = df_filtered.question_id.astype(np.int16)
        df_filtered.bundle_id = df_filtered.bundle_id.astype(np.int16)
        df_filtered.lecture_id = df_filtered.lecture_id.astype(np.int32)

        # Prefix part names with the originating dataframe name.
        df_filtered.rename(columns={'part_x': 'lecture_part',
                                    'part_y': 'question_part'},
                           inplace=True)
        
        # Save the merged.
        df_filtered.to_csv('riiid_data.csv', index=False)

        # Return the dataset.
        return df_filtered


def sampled_users(df):
    '''
    This function accepts data from `train.csv` and
    returns a random sample of 100_000 user_ids.
    '''
    user_ids = df['user_id'].value_counts()[df['user_id'].value_counts() > 10].index.to_list()
    sampled_ids = random.sample(user_ids, 100_000)
    return sampled_ids


def datatype_converter():
    '''
    This function returns a dictionary of column names and data types to convert.
    '''
    train_data_types_dict = {
    'timestamp': np.int64,
    'user_id': np.int32,
    'content_id': np.int16,
    'content_type_id': np.int16,
    'task_container_id' : np.int16,
    'user_answer' : np.int8,
    'answered_correctly': np.int8,
    'prior_question_elapsed_time': np.float16
    }
    
    lectures_data_types_dict = {
    'lecture_id' : np.int16,
    'tag' : np.int8,
    'part' : np.int8
    }

    questions_data_types_dict = {
    'question_id' : np.int16,
    'bundle_id' : np.int16,
    'part' : np.int8
    }
    
    return train_data_types_dict, lectures_data_types_dict, questions_data_types_dict


def sampled_train():
    '''
    This function selects a random sample of 100_000 users from the `train.csv` dataset.
    Returns a dataframe of 100_000 users that have more than 10 rows of data.
    
    
    Parameters
    ----------
    None
    
    Returns
    -------
    sampled_data : pandas.core.frame.DataFrame
        A pandas dataframe of 100,000 randomly selected
        users.
    '''
    train_dtypes, _, _ = datatype_converter()
    
    if os.path.isfile('sampled_train.csv'):
        # Return the cached file
        return pd.read_csv('sampled_train.csv', index_col=False, dtype=train_dtypes)
    else:   
        # Load `train.csv` data
        df = pd.read_csv('train.csv', dtype=train_dtypes, usecols=[1,2,3,4,5,6,7,8,9])

        # Randomly select 50_000 users
        sampled_ids = sampled_users(df)

        # Filter the dataframe for users
        sampled_data = df.loc[df['user_id'].isin(sampled_ids)]

        # Cache local file of sampled data.
        sampled_data.to_csv('sampled_train.csv', index=False)
    
        # Return the dataframe
        return sampled_data