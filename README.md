# :label: Plain and gentle language for more comfortable co-existence: Plain Japanese Standard
This project is conducted as a part of Bachelorarbeit in the winter semester 2022/2023.

[[_TOC_]]

## :pushpin: About the project
The aim of this project is to conduct classification and translation experiments on Japanese Simplified Corpus. TODO

## :runner: Getting started

### Prerequisites
TODO requirements

### Fine-tuned models
TODO (google drive?)

## :open_file_folder: Directory structure

```
|--dataset_analysis
|  |--analysis.py
|  |--criteria.py
|--classification
|  |--create_data_classification.py
|  |--train_classify.py
|  |--param_tune.py
|  |--train_param_tune.py
|  |--resources
|      |--df_test.csv
|      |--df_test2.csv
|      |--df_train.csv
|      |--df_train2.csv
|      |--df_valid.csv
|      |--df_valid2.csv
|--translation
|  |--create_data_translate.py
|  |--train_translate.py
|  |--evaluate.py
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

* `classification`: code of classification experiment.
* `translation`: code of translation experiment.
* `resources`: dataset for classification and translation experiments. Both same dataset, but processed differently. (See `create_data_*.py` for further details.) 
* `README.me`: this file


## :clipboard: Usage

### Classification
For fine-tuning or test models:
* TODO

For conducting hyperparameter tuning:
* TODO

### Translation
For fine-tuning or test models:
* TODO (below is dummy)

For evaluating a specific model that generates rationales with one of the supported metrics, do the following:
1. activate the neccessary virtual environment for the metric
2. run `python3 metric_compare.py` with the appropriate parameters

For example, to evaluate the "UNIFORM, VisComet text inferences" model from the Marasovic dataset with the METEOR metric, use:

`python3 metric_compare.py --model text_objects_q_a_to_r --metric METEOR --gen data/marasovic_text.csv --vcr data/val_np.jsonl`

Some notes on the various parameters:
* `--metric`: This specifies the metric to use to score the rationales. Supported metrics are: `bleu` for BLEU, `meteor` for METEOR, `bertscore` for BERTScore, 

:name_badge: Author
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