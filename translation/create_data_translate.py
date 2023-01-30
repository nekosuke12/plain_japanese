# script to create dataset

from datasets import load_dataset
from datasets import DatasetDict

raw_datasets = load_dataset("snow_simplified_japanese_corpus")

# split datasets into train/val/test 80-20-20
train_testvalid = raw_datasets["train"].train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather all to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

train_test_valid_dataset.save_to_disk("resources/train_test_valid_dataset") # save dataset to resources