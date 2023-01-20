from transformers import DataCollatorWithPadding, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric
from transformers import TrainingArguments, Trainer
import numpy as np
from ray import tune


def compute_metrics_fn(eval_preds):
    # define compute metrics
    metrics = dict()
    
    accuracy_metric = load_metric('accuracy')
    precision_metric = load_metric('precision')
    recall_metric = load_metric('recall')
    f1_metric = load_metric('f1')


    logits = eval_preds.predictions
    labels = eval_preds.label_ids
    preds = np.argmax(logits, axis=-1)  
    
    metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
    metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
    metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
    metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))


    return metrics

def train(data, model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, return_dict=True)

    # tokenize the data
    tokenized_data = data.map(preprocess_function, batched=True)


    # train mode based on framework
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # define configuration for hyperparameter tuning
    tune_config = {
        "weight_decay": tune.uniform(0, 0.02),
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "per_device_train_batch_size": tune.choice([8, 16, 32]),
        "num_train_epochs": 25
    }

    training_args = TrainingArguments(
        output_dir='tuning_roberta',
        report_to='wandb',  # Turn on Weights & Biases logging
        num_train_epochs = 10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate = 0.01, 
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        disable_tqdm=True,
        save_total_limit=1
    )

    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["valid"],
        model_init=model_init,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator,
    )  


    # Default objective is the sum of all metrics
    trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        hp_space = lambda _: tune_config,
        n_trials=10,
        name="tune_transformer_pbt",
    )

