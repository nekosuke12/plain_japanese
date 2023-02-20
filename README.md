# :label: Plain and gentle language for more comfortable co-existence: Plain Japanese Standard
This project is conducted for my bachelor thesis in the winter semester 2022/2023.

## :pushpin: About the project
The aim of this project is to conduct classification and translation experiments on Japanese Simplified Corpus **SNOW T15: やさしい日本語コーパス**. <br>
<br>
The goal of the two tasks are to answer the questions:
1. "Can a language model tell the readability of a sentence?", i.e. binary classification of original (0) or simplified (1)
2. "Can a language model learn to simplify a sentence as a translation between the same language?", i.e. simplification as a form of translation between the same language.

## :runner: Getting started

### Prerequisites

For this project, Python 3.10.9 is used.
Use requirements.txt to install necessary libraries. e.g. run `pip install -r requirements.txt` <br> 
(:warning: As `requirements.txt` is created using the conda environment I used locally for this project, it includes a large number of libraries)

### Fine-tuned models
A fine-tuned model is available for each task. [Fine-tuned Models](https://heibox.uni-heidelberg.de/d/7da9aeaeaaac4d988123/)

* fine-tuned roberta base model for classification: roberta_base.zip
* fine-tuned bart base model for translation: bart_base.zip :warning: this data is very large (1GB)

## :open_file_folder: Directory structure

```
|--dataset_analysis
|  |--criteria.py
|  |--create_vocab.py
|  |--data_analysis.py
|  |--data_analysis.ipynb
|  |--resources
|  |--analysed
|--classification
|  |--create_data_classification.py
|  |--train_classify.py
|  |--param_tune.py
|  |--train_param_tune.py
|  |--resources
|      |--df_test2.csv
|      |--df_train2.csv
|      |--df_valid2.csv
|--translation
|  |--create_data_translate.py
|  |--train_translate.py
|  |--test_translate.py
|  |--evaluate_simplified.py
|  |--resources
|      |--train_test_valid_dataset
|      |--sample_generated.txt
|  |--scores
|      |--base_scores.csv
|      |--large_scores.csv
|      |--summary_scores.ipynb
|--.gitignore
|--README.md
|--index.xml
```

Some important directories and files are:

* `analysis`: code of dataset analysis and results (the results themselves are in `analyzed` and `data_analysis_results.ipynb` contains the visualisation).
* `classification`: code of classification experiment.
* `translation`: code of translation experiment. `score` contains results.
* `resources`: dataset for classification and translation experiments. Both same dataset, but processed differently. `df_*2.csv` is named like this because they are datasets that don't contain duplicates. (See `create_data_*.py` for further details.) 
* `README.me`: this file


## :clipboard: Usage

Use GPU to run all the scripts, except for the evaluation of generated simplification (`evaluate_simplified.py`). Activate the neccessary virtual environment for the metric.

### Classification
For fine-tuning or test models:
* run `python train_classify.py` with the appropriate parameters

e.g. <br>
`python train_classify.py --mode train --checkpoint ku-nlp/roberta-base-japanese-char-wwm --output models/roberta_base` to train a roberta base model and save the model in `models/roberta_base`.<br>

`python train_classify.py --mode test --checkpoint /home/students/udaka/bachelorarbeit/classification/models/roberta_base --tokenizer ku-nlp/roberta-base-japanese-char-wwm` to test a fine-tuned roberta base model using bert base tokenizer. (replace checkpoint path with the path in which you save fine-tuned models. See Fine-tuned models section for available fine-tuned models)
<br>
<br>
For conducting hyperparameter tuning:
* run `python param_tune.py` with the appropriate parameters

e.g. `python param_tune.py --checkpoint ku-nlp/roberta-base-japanese-char-wwm` to do hyperparameter tuning on a roberta base model.

### Translation
For fine-tuning or test models:
* run `python train_translate.py` with the appropriate parameters

e.g. <br>
`python train_translate.py --checkpoint Formzu/bart-base-japanese --output models/bart_base` to train a bart base model and save the model in `models/bart_base`. <br>

`python test_translate.py --checkpoint /home/students/udaka/bachelorarbeit/translation/models/bart_base --tokenizer Formzu/bart-large-japanese --output bart_base.txt` to test a bart base  model and output the generated test in bart_base.txt. (replace checkpoint path with the path in which you save fine-tuned models. See Fine-tuned models section for available fine-tuned models)

<br>
<br>
For evaluating the quality of generated simplification:
* run `python evaluate_simplified.py` with the appropriate parameters

e.g. `python evaluate_simplified.py --generated bart_base.txt`  to evaluate bart_base.txt.


## :name_badge: Author
- Maya Udaka

udaka@cl.uni-heidelberg.de

## :link: References
### Dataset
- [言語商会: SNOW T15:やさしい日本語コーパス](https://www.jnlp.org/GengoHouse/snow/t15)
- [huggingface: Datasets: snow_simplified_japanese_corpus](https://huggingface.co/datasets/snow_simplified_japanese_corpus)

### Models
#### classification
- [BERT base Japanese model](https://huggingface.co/cl-tohoku/bert-base-japanese)
- [RoBERTa base Japanese model](https://huggingface.co/ku-nlp/roberta-base-japanese-char-wwm)
- [LUKE base Japanese model](https://huggingface.co/studio-ousia/luke-japanese-base-lite)
#### translation
- [BART base Japanese model](https://huggingface.co/Formzu/bart-base-japanese)
- [BART large Japanese model](https://huggingface.co/Formzu/bart-large-japanese)

### Metrics 
- [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu)
- [BERTscore](https://huggingface.co/spaces/evaluate-metric/bertscore)
- [SARI](https://huggingface.co/spaces/evaluate-metric/sari)


### Others
- [huggingface: classification demo](https://github.com/huggingface/notebooks/blob/6ca682955173cc9d36ffa431ddda505a048cbe80/examples/text_classification.ipynb)
- [huggingface: translation demo](https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb)
- [mecab-python3](https://github.com/SamuraiT/mecab-python3)
