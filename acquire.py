# Acquire File
# Imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os

###################### Acquire Riiid Data ########################
def get_riiid_data():
    # Acquire data
    df_train = pd.read_csv('train.csv')
    df_lectures = pd.read_csv('lectures.csv')
    df_questions = pd.read_csv('questions.csv')

    return df