import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
    print(user_ids)
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
    return train, test