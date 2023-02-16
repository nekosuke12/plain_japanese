"""
@project: Plain Japanese
@author:  Maya Udaka
udaka@cl.uni-heidelberg.de
@filename: data_analysis.py
@description: analyse dataset based on each linguistic attribute
"""

# Due to the error currently occurring during the installation of sudachidict_core, the script may not run properly

import spacy
import pandas as pd
import criteria
import csv
import argparse

def evaluate(text, nlp):
    # Duplikate z.B. で muss als Ausnahme　
    doc = nlp(text)

    diff_words = criteria.check_difficult_words(doc, nlp)
    loan_words = criteria.check_loan_words(doc, nlp)
    teineigo = criteria.is_teineigo(doc, nlp)
    doubleneg = criteria.check_doubleneg(doc, nlp)
    causative = criteria.check_causative(doc, nlp)
    bunsetsu = criteria.check_bunsetsu(doc)
    beats = criteria.check_beats(doc)
    kanjis = criteria.check_kanjis(doc)

    result = [diff_words, loan_words, teineigo, doubleneg, causative, bunsetsu, beats, kanjis]
    return  result
    



if __name__ == '__main__':
    nlp = spacy.load("ja_ginza")

    parser = argparse.ArgumentParser(description='Analyse a dataset')

    parser.add_argument('-i', '--input', help='input', required=True) # e.g. resources/test_original.txt, resources/test_simplfiied.txt
    parser.add_argument('-o', '--output', help='output path', required=True) # e.g. test_original_analysed.csv, test_dimplified_analysed
    
    args = parser.parse_args()
    input = str(args.input)
    output_path = str(args.output)

    if not input.endswith(".txt") and len(input) > 1:
        input_text = input.replace("。", "")

        if input_text.count("。") >= 2: # if there are multiple sentences in the input
            splited_text = input_text.split("。")
            for sent in splited_text:
                if sent != "":
                    print(evaluate(sent, nlp)[-1])
        else: # if there is only one sentence in the input
            print("Result is ", (evaluate(input_text, nlp)))

    elif input.endswith(".txt"): # when a .txt is loaded
        header = ["diff_words", "loan_words", "teineigo", "doubleneg", "causative", "bunsetsu", "beats", "kanjis"]

        with open(input, "r") as textfile, open(output_path, 'w') as myfile:
            writer=csv.writer(myfile, delimiter='\t', lineterminator='\n')
            writer.writerow(header)

            for line in textfile:
                writer.writerow(evaluate(line, nlp))


    else:
        print('invalid input!')