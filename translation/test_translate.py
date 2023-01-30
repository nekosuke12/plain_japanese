from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from datasets import load_from_disk
import pandas as pd
import argparse


if __name__ == '__main__':
    # in command line, e.g. python test_translate.py --checkpoint Formzu/bart-large-japanese --tokenizer Formzu/bart-large-japanese --output simplified/bart_base.txt
    # parse the arguments
    parser = argparse.ArgumentParser(description='Test a model for translation')

    parser.add_argument('-c', '--checkpoint', help='use checkpoint of your choice', required=True)
    parser.add_argument('-t', '--tokenizer', help='use tokenizer of your choice, only for test mode', required=True)
    parser.add_argument('-o', '--output', help='output path', required=False)
    
    args = parser.parse_args()
    model_checkpoint = str(args.checkpoint)
    tokenizer_path = str(args.tokenizer)
    output_path = str(args.output)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    translator = pipeline("translation", model=model_checkpoint, tokenizer=tokenizer)


    train_test_valid_dataset = load_from_disk("resources/train_test_valid_dataset")
    test_dataset = train_test_valid_dataset["test"]
    df_test = pd.DataFrame.from_dict(test_dataset)

    test_texts = df_test["original_ja"].tolist()

    with open(output_path, 'w') as myfile:
        for line in test_texts:
            translation = translator(line, max_length=100)
            myfile.write(translation[0]['translation_text'])
            myfile.write('\n')
