"""
@project: Plain Japanese
@author:  Maya Udaka
udaka@cl.uni-heidelberg.de
@filename: create_data_classification.py
@description: script to adjust the original dataset to classification task
"""

import pandas as pd
from datasets import load_dataset,DatasetDict
import numpy as np

def modify_df(df): 
    # make original dataset suitable for classification (text and label)
    print("overall: ", len(df))
    to_exclude_from_org = np.where(df['original_ja'] == df['simplified_ja']) # newly added
    print("overlapped: ", len(to_exclude_from_org[0])) # the number of overlapped (original/simplified) data

    df_org = df.drop(['original_en', 'simplified_ja', 'ID'], axis=1)
    df_org = df_org.drop(to_exclude_from_org[0]) # newly added
    df_org['label'] = 0

    df_org = df_org.rename(columns={'original_ja': 'text'})

    df_sim = df[['simplified_ja']].copy()
    df_sim['label'] = 1

    df_sim = df_sim.rename(columns={'simplified_ja': 'text'})

    frames = [df_org, df_sim]

    result = pd.concat(frames, ignore_index=True).sample(frac = 1)
    return result


raw_datasets = load_dataset("snow_simplified_japanese_corpus")

# split datasets into train/val/test 80-10-10
train_testvalid = raw_datasets["train"].train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

# gather all to create a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


# pull data into pandas dataframes
df_train = pd.DataFrame.from_dict(train_test_valid_dataset['train'])
df_train = modify_df(df_train)
df_train.to_csv('resources/df_train2.csv', sep='\t') # change path to save

df_valid = pd.DataFrame.from_dict(train_test_valid_dataset['valid'])
df_valid = modify_df(df_valid)
df_valid.to_csv('resources/df_valid2.csv', sep='\t') # change path to save

df_test = pd.DataFrame.from_dict(train_test_valid_dataset['test'])
df_test = modify_df(df_test)
df_test.to_csv('resources/df_test2.csv', sep='\t') # change path to save

