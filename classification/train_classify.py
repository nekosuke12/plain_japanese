# script to train models for classification
# reference: https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb

import pandas as pd
import numpy as np
import sys
import evaluate
from datasets import Dataset, DatasetDict, Value, ClassLabel, Features
from transformers import DataCollatorWithPadding, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import argparse


def create_dataset():
    # load dataset created using create_data_classification.py
    df_train = pd.read_csv('resources/df_train2.csv', sep='\t', header=0) # # change path to save 
    df_valid = pd.read_csv('resources/df_valid2.csv', sep='\t', header=0) 
    df_test = pd.read_csv('resources/df_test2.csv', sep='\t', header=0)

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

def train(data, model_checkpoint, output_path):
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 100
    weight_decay = 0.01 

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # tokenize the data
    tokenized_data = data.map(preprocess_function, batched=True)

        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )    

    trainer.train()
    trainer.save_model(output_path)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='Fine-tuning a model for classification')

    parser.add_argument('-m', '--mode', help='choose train or test mode', required=True)
    parser.add_argument('-c', '--checkpoint', help='use checkpoint of your choice', required=True)
    parser.add_argument('-t', '--tokenizer', help='use tokenizer of your choice, only for test mode', required=False)
    parser.add_argument('-o', '--output', help='trained model output path', required=False)
    
    args = parser.parse_args()
    model_checkpoint = str(args.checkpoint)
    tokenizer = str(args.tokenizer)
    output_path = str(args.output)


    # create the dataset
    data = create_dataset()
    
    # check the content of dataset
    print(data['train'].features)
    print(data['valid'].features)
    print(data)

    # in command line, e.g. python train_classify.py --mode train --checkpoint ku-nlp/roberta-base-japanese-char-wwm --output models/roberta_base
    if args.mode == "train":
        train(data, model_checkpoint, output_path)

    # in command line, e.g. python train_classify.py --mode test --checkpoint /home/students/udaka/bachelorarbeit/classification/models/roberta_base --tokenizer ku-nlp/roberta-base-japanese-char-wwm
    if args.mode == "test":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
        )
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        

        tokenized_data = data.map(preprocess_function, batched=True)


        model.eval() 

        # code to calculate f1 score
        predicted = trainer.predict(tokenized_data["test"])
        
        pred_labels = []
        for arr in predicted[0]:
            pred_labels.append(np.argmax(arr))


        org_labels = predicted[1]

        print(predicted)
        print("f1 score", f1_score(org_labels, pred_labels))

    else:
        print("error")