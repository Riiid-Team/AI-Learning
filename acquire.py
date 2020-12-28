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
    Returns a dataframe.

    Parameters
    ----------
    None 
    
    Returns
    -------
    df_data : pandas.core.frame.DataFrame
        The merged dataset of `train.csv`, `lectures.csv`, and  `questions.csv`
    '''

    if os.path.isfile('riiid_data.csv'):
        return pd.read_csv('riiid_data.csv')
    else:
        # Dictionaries to cast column dtypes.
        train_dtypes, lecture_dtypes, question_dtypes = datatype_converter()
        
        # Load each dataset
        df_train = pd.read_csv('train.csv', dtype=train_dtypes, usecols=[1,2,3,4,5,6,7,8,9])
        df_lectures = pd.read_csv('lectures.csv', dtype=lecture_dtypes)
        df_questions = pd.read_csv('questions.csv', dtype=question_dtypes)
        
        # Left join df_train and df_lectures using `content_id` as the primary key.
        df_merged = df_train.merge(df_lectures, left_on='content_id', right_on='lecture_id', how='left')
        
        # Left join df_merged and df_questions using `content_id` as the primary key.
        df_data = df_merged.merge(df_questions, left_on='content_id', right_on='question_id', how='left')
        
        # Cast the data types of numeric columns.
        df_data.lecture_id = df_data.lecture_id.astype('Int16')
        df_data.tag = df_data.tag.astype('Int8')
        df_data.part_x = df_data.part_x.astype('Int8')
        df_data.part_y = df_data.part_y.astype('Int8')
        df_data.question_id = df_data.question_id.astype('Int16')
        df_data.bundle_id = df_data.bundle_id.astype('Int16')
        df_data.lecture_id = df_data.lecture_id.astype('Int32')

        # Prefix part names with the originating dataframe name.
        df_data.rename(columns={'part_x': 'lecture_part',
                                'part_y': 'question_part'},
                       inplace=True)
        
        # Cache the merged dataframe.
        df_data.to_csv('riiid_data.csv', index=False)

        # Return the dataset.
        return df_data


def datatype_converter():
    '''
    This function returns three dictionaries of column names in
    train.csv, lectures.csv, and questions.csv with data types to convert.
    '''
    # Column names in train.csv
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
    
    # Column names in lectures.csv
    lectures_data_types_dict = {
        'lecture_id' : np.int16,
        'tag' : np.int8,
        'part' : np.int8
    }

    # Column names in questions.csv
    questions_data_types_dict = {
        'question_id' : np.int16,
        'bundle_id' : np.int16,
        'part' : np.int8
    }
    
    return train_data_types_dict, lectures_data_types_dict, questions_data_types_dict


###################### Create a sampled dataset of train.csv ########################
def sampled_riiid_data(n=100_000):
    '''
    This function selects a random sample of 100_000 users from the `train.csv` dataset.
    Returns a dataframe of data from 100_000 users.
    
    
    Parameters
    ----------
    n : integer, default 100_000
        The number of unique users to sample from the dataset.
    
    Returns
    -------
    sampled_data : pandas.core.frame.DataFrame
        A pandas dataframe of 100,000 randomly selected users.
    '''
    # Load dtype conversions
    train_dtypes, lecture_dtypes, question_dtypes = datatype_converter()
    
    # Load dataset and convert column dtypes.
    df_train = pd.read_csv('train.csv', dtype=train_dtypes, usecols=[1,2,3,4,5,6,7,8,9])
    df_lectures = pd.read_csv('lectures.csv', dtype=lecture_dtypes)
    df_questions = pd.read_csv('questions.csv', dtype=question_dtypes)


    # Randomly select 100_000 users
    sampled_ids = sampled_users(df_train, sample_size=n)

    # Filter the dataframe for users
    df_sampled = df_train.loc[df_train['user_id'].isin(sampled_ids)]

    
    # Left join df_train and df_lectures using `content_id` as the primary key.
    df_merged = df_sampled.merge(df_lectures, left_on='content_id', right_on='lecture_id', how='left')

    # Left join df_merged and df_questions using `content_id` as the primary key.
    df_data = df_merged.merge(df_questions, left_on='content_id', right_on='question_id', how='left')

    # Cast the data types of numeric columns.
    df_data.lecture_id = df_data.lecture_id.astype('Int16')
    df_data.tag = df_data.tag.astype('Int8')
    df_data.part_x = df_data.part_x.astype('Int8')
    df_data.part_y = df_data.part_y.astype('Int8')
    df_data.question_id = df_data.question_id.astype('Int16')
    df_data.bundle_id = df_data.bundle_id.astype('Int16')
    df_data.lecture_id = df_data.lecture_id.astype('Int32')

    # Prefix part names with the originating dataframe name.
    df_data.rename(columns={'part_x': 'lecture_part',
                            'part_y': 'question_part'},
                   inplace=True)

    # Cache the merged dataframe.
    df_data.to_csv('sampled_riiid_data.csv', index=False)

    # Return the dataset.
    return df_data



def sampled_users(df, sample_size=100_000):
    '''
    This function accepts data from `train.csv` and
    returns a list of 100_000 randomly sampled user ids.
    '''
    # Set a random seed for reproducibility.
    random.seed(123)
    
    # user_ids with 10 or more interactions
    user_ids = df.user_id.value_counts()[df.user_id.value_counts() > 9].index.tolist()
    
    # Select a random sample of 100_000 users
    sampled_ids = random.sample(user_ids, sample_size)
    return sampled_ids
