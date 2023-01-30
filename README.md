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
|      |--df_test2.csv
|      |--df_train2.csv
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

Use GPU to run all the scripts, except for the evaluation of generated simplification (`evaluate_simplified.py`). Activate the neccessary virtual environment for the metric.

### Classification
For fine-tuning or test models:
* run `python train_classify.py` with the appropriate parameters

e.g. <br>
`python train_classify.py --mode train --checkpoint cl-tohoku/bert-base-japanese` to train a bert base model.<br>

`python train_classify.py --mode test --checkpoint cl-tohoku/bert-base-japanese --tokenizer cl-tohoku/bert-base-japanese` to test a bert base model using bert base tokenizer. 
<br>
<br>
For conducting hyperparameter tuning:
* run `python param_tune.py` with the appropriate parameters

e.g. `python param_tune.py --checkpoint ku-nlp/roberta-base-japanese-char-wwm` to do hyperparameter tuning on a roberta base model.

### Translation
For fine-tuning or test models:
* run `python train_translate.py` with the appropriate parameters

e.g. <br>
`python train_translate.py --checkpoint Formzu/bart-base-japanese --output models/translate_base` to train a bart base model and output the model in models/translate_base. <br>

e.g. `python test_translate.py --checkpoint Formzu/bart-base-japanese --tokenizer Formzu/bart-large-japanese --output simplified/bart_base.txt` to test a bart base  model and output the generated test in simplified/bart_base.txt.

<br>
<br>
For evaluating the quality of generated simplification:
* run `python evaluate_simplified.py` with the appropriate parameters

e.g. `python evaluate_simplified.py --generated bart_base.txt`  to evaluate bart_base.txt.


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