# Acquire File
# Imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os

###################### Acquire Riiid Data ########################

def get_riiid_data():
    '''
    This function acquires `train`, `lectures`, and `questions` datasets and
    returns a merged spark dataframe.
    
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
        train_data_types_dict = {
        'timestamp': np.int64,
        'user_id': np.int32,
        'content_id': np.int16,
        'content_type_id': np.int16,
        'task_container_id' : np.int16,
        'user_answer' : np.int8,
        'answered_correctly': np.int8,
        'prior_question_elapsed_time': np.float16,
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
        
        # Acquire data
        df_train = pd.read_csv('train.csv', dtype=train_data_types_dict, nrows=40e6)
        
        # Remove the last user missing data.
        df_train = df_train.loc[df_train.user_id != df_train.user_id.max()]
        
        df_lectures = pd.read_csv('lectures.csv', dtype=lectures_data_types_dict)
        df_questions = pd.read_csv('questions.csv', dtype=questions_data_types_dict)
        
        # Left join df_train and df_lectures using `content_id` as the primary key.
        df_merged = df_train.merge(df_lectures, left_on='content_id', right_on='lecture_id', how='left')
        
        # Left join df_merged and df_questions using `content_id` as the primary key.
        df = df_merged.merge(df_questions, left_on='content_id', right_on='question_id', how='left')
    
        # Fill in missing values with 0.
        df_filtered = df.fillna(0)
        
        # Change the data types numeric columns.
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
        
        # Return the merged dataset.
        return df_filtered