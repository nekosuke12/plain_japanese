# -*- coding: utf-8 -*-

import sys
from evaluate import load
import pandas as pd
from datasets import load_from_disk
import MeCab


# functions for metrics
def calculate_sari(sources, predictions, references):
    sari = load("sari")
    return sari.compute(sources=sources, predictions=predictions, references=references)

def calculate_bert(predictions, references):
    bertscore = load("bertscore")
    return bertscore.compute(predictions=predictions, references=references, lang="jp")

def calculate_bleu(predictions, references, tagger):
    # tokenize the input texts first

    tag_pred = [tagger.parse(str(predictions))]
    tag_ref = [[tagger.parse(str(references).replace('\n', ''))]]
    bleu = load("bleu")
    return bleu.compute(predictions=tag_pred , references=tag_ref)



if __name__ == '__main__':
    # TODO argparse?
    # load tagger for bleu: please download unidic locally if this path does not work 
    tagger = MeCab.Tagger('-r/home/students/udaka/miniconda3/envs/JPJ/lib/python3.9/site-packages/unidic_lite/dicdir/mecabrc -d/home/students/udaka/miniconda3/envs/JPJ/lib/python3.9/site-packages/unidic_lite/dicdir/  -Owakati')
    
    # load generated data
    generated_text = sys.argv[1]
    generated = []
    with open(generated_text, 'r') as r:
        for line in r:
            generated.append(line.rstrip().replace(' ', '')) # remove whitespace


    # load test data
    train_test_valid_dataset = load_from_disk("resources/train_test_valid_dataset")
    test_dataset = train_test_valid_dataset["test"]
    df_test = pd.DataFrame.from_dict(test_dataset)

    test_plang = df_test["simplified_ja"].tolist() # simplified
    test_org = df_test["original_ja"].tolist() # original


    sari_scores = []
    bleu_scores = []
    bert_scores_pre = []
    bert_scores_rec = []
    bert_scores_f1 = []

    for gen, plang, org in zip(generated, test_plang, test_org):
        print(gen, "\n")
        print(plang, "\n")
        print(org, "\n")
        sources = [org]
        predictions = [gen]
        references = [[plang]]

        if plang != org:
        
            sari_score = calculate_sari(sources, predictions, references)['sari']
            print(sari_score)
            sari_scores.append(sari_score)
            
            bleu_score = calculate_bleu(predictions, references, tagger)['bleu']
            print(bleu_score)
            bleu_scores.append(bleu_score)

            bert_score = calculate_bert(predictions, references)
            print(bert_score)
            bert_scores_pre.append(bert_score['precision'][0])
            bert_scores_rec.append(bert_score['recall'][0])
            bert_scores_f1.append(bert_score['f1'][0])
        
        # if plang and org are identical, the score should be nan
        else:
            sari_scores.append('nan')
            bleu_scores.append('nan')
            bert_scores_pre.append('nan')
            bert_scores_rec.append('nan')
            bert_scores_f1.append('nan')


    list_of_tuples = list(zip(test_org, test_plang, generated, sari_scores, bleu_scores, bert_scores_f1,  bert_scores_pre, bert_scores_rec))

    df = pd.DataFrame(list_of_tuples,
                  columns=['original', 'simplified', 'generated','sari', 'bleu', 'bert_f1', 'bert_pre', 'bert_rec'])

    df.to_csv('scores/base_scores.csv', index=False) # save calculated scores
    # See "bachelorarbeit/translation/scores" for the saved data
