# load the model

import pandas as pd
from datasets import Dataset, DatasetDict, Value, ClassLabel, Features
from ray.dashboard import *
import train_param_tune

# load the dataset in resources
def create_dataset():
    # load data
    df_train = pd.read_csv('resources/df_train2.csv', sep='\t', header=0)  
    df_valid = pd.read_csv('resources/df_valid2.csv', sep='\t', header=0) 
    df_test = pd.read_csv('resources/df_test2.csv', sep='\t', header=0)

    # check data
    print(df_train.head())

    # define the features
    features = Features({
        "text": Value("string"), 
        "label": ClassLabel(num_classes=2, names=['org','plang']),
        "Unnamed: 0": Value("string") 
        })

    # recreate the data object using the smaller df's
    data = DatasetDict({
        'train': Dataset.from_pandas(df_train, features=features),
        'valid': Dataset.from_pandas(df_valid, features=features),
        'test': Dataset.from_pandas(df_test, features=features),
        })

    # remove index col (seems to be coming in from pandas for some reason)
    data = data.remove_columns(["Unnamed: 0"])

    return data




if __name__ == "__main__":
    # TODO argparse
    data = create_dataset()
    model_checkpoint = 'ku-nlp/roberta-base-japanese-char-wwm'
    train_param_tune.train(data, model_checkpoint)
