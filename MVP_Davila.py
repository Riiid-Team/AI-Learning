import pandas as pd
import numpy as np


def sam_train_features(df):
    
    # average user accuracy
    user_acc_mean = pd.DataFrame(df.groupby('user_id')['answered_correctly'].mean())
    user_acc_mean.columns = ['user_acc_mean']
    df = df.merge(user_acc_mean, how = 'left', left_on = 'user_id', right_on = 'user_id')
    
    # running count of lectures viewed per user
    df['user_lectures_running_total'] = df.groupby(by=['user_id'])['content_type_id'].transform(lambda x: x.cumsum())
    
    # time taken to answer previous question
    lastq = pd.DataFrame(df.timestamp.diff())

    lastq.columns = ['last_q_time']

    lastq.fillna(0)

    df = pd.concat([df, lastq], axis=1)
    
    df['last_q_time'] = np.where((df.last_q_time.isnull()), 0, df.last_q_time)
    df['last_q_time'] = np.where((df.last_q_time < 0), 0, df.last_q_time)
    
    for x in range(0,5):
    # set to loop number of times == largest count of questions in bundle
        df['last_q_time'] = np.where((df.last_q_time ==  0) & (df.prior_question_elapsed_time.isnull() != True), df.last_q_time.shift(1), df.last_q_time)
    
    # avg time each user takes a question
    avg_q_time_user = pd.DataFrame(df.groupby('user_id').mean().round()['last_q_time'])
    
    avg_q_time_user.columns = ['avg_user_q_time']
    
    df = df.merge(avg_q_time_user, how='left', left_on='user_id', right_on='user_id')
    
    df.drop(columns='last_q_time', inplace=True)

    return df

def sam_valtest_features(df, val_or_test):
    
    # average user accuracy
    user_acc_mean = pd.DataFrame(df.groupby('user_id')['answered_correctly'].mean())
    user_acc_mean.columns = ['user_acc_mean']
    val_or_test = val_or_test.merge(user_acc_mean, how = 'left', left_on = 'user_id', right_on = 'user_id')
    
    # running count of lectures viewed per user
    val_or_test['user_lectures_running_total'] = val_or_test.groupby(by=['user_id'])['content_type_id'].transform(lambda x: x.cumsum())
    
    # avg time each user takes a question
    avg_q_time_user = pd.DataFrame(df.groupby('user_id').mean().round()['last_q_time'])
    
    avg_q_time_user.columns = ['avg_user_q_time']
    
    val_or_test = val_or_test.merge(avg_q_time_user, how='left', left_on='user_id', right_on='user_id')
    
    return val_or_test