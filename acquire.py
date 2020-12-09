# Acquire File
# Imports 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os

import pyspark
from pyspark.sql import SparkSession


###################### Acquire Riiid Data ########################


def get_riiid_data():
    '''
    This function acquires `train`, `lectures`, and `questions` datasets and
    returns a merged spark dataframe.

    Parameters
    ----------
    None 
    
    Returns
    -------
    df :  pyspark.sql.dataframe.DataFrame
    '''

    spark = SparkSession.builder.getOrCreate()
    
    # Acquire data
    df_train = spark.read.csv('train.csv', header=True, inferSchema=True)
    df_lectures = spark.read.csv('lectures.csv', header=True, inferSchema=True)
    df_questions = spark.read.csv('questions.csv', header=True, inferSchema=True)
    
    # Left join df_train and df_lectures using `content_id` as the primary key.
    df_merged = df_train.join(df_lectures, df_train.content_id == df_lectures.lecture_id, how='left')
    
    # Left join df_merged and df_questions using `content_id` as the primary key.
    df = df_merged.join(df_questions, df_merged.content_id == df_questions.question_id, how='left')

    # Return the merged dataset.
    return df