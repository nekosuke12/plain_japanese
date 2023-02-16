"""
@project: Plain Japanese
@author:  Maya Udaka
udaka@cl.uni-heidelberg.de
@filename: create_vocab.py
@description: script to create a plain language vocabulary consisting of easy words
"""

import pandas as pd


def kat_to_hir(text):
    """convert hiragana to katakana."""
    # https://github.com/olsgaard/Japanese_nlp_scripts/blob/master/hiragana_katakana_translitteration.py

    katakana_chart = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヽヾ"
    hiragana_chart = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖゝゞ"
    kat2hir = str.maketrans(katakana_chart, hiragana_chart)

    return text.translate(kat2hir)

def load_JLPT_vocab(filename):
    """load vocabulary from a JLPT word list"""
    vocab = pd.read_csv(filename, encoding = "UTF-16LE", sep='\t', header=0, on_bad_lines = 'warn')
    kanji = vocab["KANJI"].tolist()
    furigana = vocab["FURIGANA"].tolist()
    word_list = kanji + furigana

    return [*set(word_list)] # to remove duplicates

def load_JP_dict_vocab():
    """load vocabulary from jp_vocab"""
    jp_vocab = pd.read_csv('resources/jp_vocab.tsv', encoding = "UTF-16LE", sep='\t', header=0, on_bad_lines = 'warn') # load the vocabulary
    easy_levels = ["1.初級前半", "2.初級後半", "3.中級前半"]

    easy_jp_vocab = jp_vocab[jp_vocab["語彙の難易度"].isin(easy_levels)]
    easy_jp_vocab_kanji = easy_jp_vocab["標準的な表記"].tolist()

    easy_jp_vocab_hiragana = [kat_to_hir(k) for k in easy_jp_vocab["読み"].tolist()]
    word_list = easy_jp_vocab_kanji + easy_jp_vocab_hiragana

    return [*set(word_list)]


def create_easy_vocab():
    """create easy vocabulary"""
    n4_words = load_JLPT_vocab('resources/jlpt-n4-nomeaning.tsv') # load n4 vocabulary
    n5_words = load_JLPT_vocab('resources/jlpt-n5-nomeaning.tsv') # load n5 vocabulary
    easy_dict_words = load_JP_dict_vocab() # load easy words from dictionary
    particles = ['ば', 'ばかり', 'だけ', 'でも', 'が', 'か', 'かしら', 'かい', 'かな', 'から', 'けど', 'けれども'] 
    # https://jlptsensei.com/complete-japanese-particles-list/
    teineigo = ['です', 'ます']

    word_list = [*set(n4_words + n5_words + easy_dict_words + particles + teineigo)]
    to_remove = ['nan', 'もっと', 'ゆっくり', 'きっと', 'しっかり', 'はっきり', 'すっかり', 'やっと', 'そろそろ', 'どんどん', 'すっと']

    return  [x for x in word_list if str(x) not in to_remove] # to remove nan

def load_katakana_vocab():
    """load foreign_words_preprocessed.txt"""
    with open("resources/foreign_words_preprocessed.txt", "r") as f:
        katakanas = f.readlines()

    return [x.strip() for x in katakanas]