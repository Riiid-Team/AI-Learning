# Libraries to evaluate our model's performance
import pandas as pd
import numpy as np

from sklearn.metric import confusion_matrix, classification_report

def data_split(df):
    '''
    
    '''
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    # Set up the train size
    train_size = 0.8
    validate_size = 0.1

    sampled_ids = df.user_id.unique()
    
    for user in sampled_ids:
        data = df.loc[df['user_id'] == user]
        n = data.shape[0]

        train_end_index = int(train_size * n)
        validate_end_index = train_end_index + int(validate_size * n)

        df_train = data.iloc[:train_end_index]
        df_validate = data.iloc[train_end_index:validate_end_index]
        df_test = data.iloc[validate_end_index:]

        train = pd.concat([train, df_train])
        validate = pd.concat([validate, df_validate])
        test = pd.concat([test, df_test])
        
        return train, validate, test