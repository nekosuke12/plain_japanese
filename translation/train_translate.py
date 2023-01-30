# script to train models for translation (simplification)
# reference: https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from datasets import load_from_disk
import numpy as np
import argparse


def train(model_checkpoint, output_path):
    # parameters for preprocessing
    max_input_length = 128
    max_target_length = 128
    source_lang = "original_ja"
    target_lang = "simplified_ja"

    # necessary functions
    def preprocess_function(examples):
        # the model selected can handle will be truncated to the maximum length accepted by the model 
        inputs = [ex for ex in examples[source_lang]]
        targets = [ex for ex in examples[target_lang]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
   
   
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

    # load datasets from ``resources" 
    train_test_valid_dataset = load_from_disk("resources/train_test_valid_dataset")
    
    # apply preprocess_function to all data
    tokenized_datasets = train_test_valid_dataset.map(preprocess_function, batched=True)


    # load metric
    metric = load_metric("sacrebleu")


    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    batch_size = 16
    model_name = model_checkpoint.split("/")[-1]

    # to instantiate trainer 
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=100,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    # pass all of this with out datasets to trainer 
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # fine-tune the model
    trainer.train()

    trainer.save_model(output_path)


if __name__ == '__main__':
    # in command line, e.g. python train_translate.py --checkpoint Formzu/bart-large-japanese --output models/translate_base

    # parse the argument
    parser = argparse.ArgumentParser(description='Fine-tune a model for translation')

    parser.add_argument('-m', '--checkpoint', help='checkpoint of your choice', required=True)
    parser.add_argument('-o', '--output', help='trained model output path', required=True)
    args = parser.parse_args()
    
    model_checkpoint = str(args.checkpoint)
    output_path = str(args.output)

    train(model_checkpoint, output_path)

 