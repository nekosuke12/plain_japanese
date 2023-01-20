import sys
import spacy
import pandas as pd
import criteria
import csv

def evaluate(text, nlp):
    # TODO
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

    # TODO argparse


    if not sys.argv[1].endswith(".txt") and len(sys.argv[1]) > 1:
        input_text = sys.argv[1].replace("。", "") # TODO remove punctuation

        if input_text.count("。") >= 2: # if there are multiple sentences in the input
            splited_text = input_text.split("。")
            for sent in splited_text:
                if sent != "":
                    print(evaluate(sent, nlp)[-1])
        else: # if there is only one sentence in the input
            print("Result is ", (evaluate(input_text, nlp)))

    elif sys.argv[1].endswith(".txt"): # when a .txt is loaded
        # TODO
        name = sys.argv[1].replace("resources/","").replace(".txt", "")
        header = ["diff_words", "loan_words", "teineigo", "doubleneg", "causative", "bunsetsu", "beats", "kanjis"]

        with open(sys.argv[1], "r") as textfile, open('analysis/%s_analysis.csv'%name, 'w') as myfile:
            writer=csv.writer(myfile, delimiter='\t', lineterminator='\n')
            writer.writerow(header)

            for line in textfile:
                writer.writerow(evaluate(line, nlp))
                 #myfile.writelines(str(evaluate(line, nlp, False)))
                 #myfile.writelines("\n")


    else:
        print('invalid input!')