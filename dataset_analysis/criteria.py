from ast import Pass
import ginza
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
import pandas as pd
import create_vocab


def check_difficult_words(doc, nlp):
    """Use simple words such as those used in Japanese-Language Proficiency Test N4 - N5 level."""

    easy_jp_vocab = create_vocab.create_easy_vocab() 

    matcher = Matcher(nlp.vocab)
    pattern = [{"LEMMA":  {"IN": easy_jp_vocab}}]
    matcher.add("easy_vocab", [pattern])
    matched = matcher(doc)
    
    easy_word_list = []
    for _, start, end in matched:
        matched_span = doc[start:end]
        easy_word_list.append(str(matched_span))
    
    diff_list = []
    for token in doc:
        if token.text not in easy_word_list:
            diff_list.append(token.text)

    return len(diff_list) / len(doc.text) # returns the percentage of difficult words


def check_loan_words(doc, nlp):
    katakanas = create_vocab.load_katakana_vocab()

    matcher = Matcher(nlp.vocab)
    pattern = [{"TEXT":  {"IN": katakanas}}]
    matcher.add("loan_words", [pattern])
    matched = matcher(doc)
    
    loan_word_list = []
    for _, start, end in matched:
        matched_span = doc[start:end]
        loan_word_list.append(str(matched_span))


    return len(loan_word_list)
    

def is_teineigo(doc, nlp):
    # judge if the sentence is teineigo. if True, a given sentence is teineigo
    # assuming that the way of speaking doesn't change within one sentence
    matcher = Matcher(nlp.vocab)

    pattern = [
        {"LEMMA": {"IN": ["ます", "です"]}, "POS": "AUX"}
    ]

    matcher.add("POLITENESS_PATTERN", [pattern])
    matched = matcher(doc)

    if len(matched) != 0:
        return 1 # teineigo
    else:
        return 0



def check_doubleneg(doc, nlp):
    # judge if negative phrase appears twice in the input sentence
    phrase_matcher = PhraseMatcher(nlp.vocab)

    neg = [nlp.make_doc(text) for text in ['ない', 'なく', 'なかった']]
    neg2 = [nlp.make_doc(text) for text in ['ありません']]

    phrase_matcher.add("NEG",None, *neg)
    phrase_matcher.add("NEG2",None, *neg2)

    matched = phrase_matcher(doc)

    if len(matched) != 0:
        return 1 # contains doubleneg
    else:
        return 0

def check_causative(doc, nlp):
    # check if a sentence contains causative
    causative = ['させる', 'せる']

    matcher = Matcher(nlp.vocab)
    pattern = [{"LEMMA":  {"IN": causative}}]
    matcher.add("easy_vocab", [pattern])
    matched = matcher(doc)

    if len(matched) != 0:
        return 1 # teineigo
    else:
        return 0


def check_bunsetsu(doc):
    # judge if a sentence has not too many bunsetsu
    num_span = 0
    for sent in doc.sents:
        for span in ginza.bunsetu_spans(sent):
            num_span += 1
    return num_span


def check_beats(doc): # mora
    text_in_hiragana = []
    for sent in doc.sents:
        for token in sent:
            text_in_hiragana.append(token.morph.get("Reading"))

    text_in_hiragana = ''.join([''.join(x) for x in text_in_hiragana]) # convert from list of lists to list and then to a string
    
    return len(text_in_hiragana)


def check_kanjis(doc):
    # check if there are not too many kanjis
    def is_hiragana(x):
        # judge if a given input is hiragana using unicode block
        # https://de.wikipedia.org/wiki/Unicodeblock_Hiragana
        return "ぁ" <= x <= "ゔ"


    def is_katakana(x):
        # judge if a given input is katakana using unicode block
        # https://en.wikipedia.org/wiki/Katakana_(Unicode_block)
        return "゠" <= x <= "ヿ"

    kanji_total = 0

    for token in doc:
        for letter in token.text:
            if not (is_hiragana(letter) or is_katakana(letter)): # i.e. kanji
                kanji_total += 1
    return kanji_total / len(doc.text) 