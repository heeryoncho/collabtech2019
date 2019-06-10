import os
import pandas as pd
import MeCab
from konlpy.tag import Mecab
import re
import pickle
from collections import Counter

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code tokenizes the J-pop/K-pop lyrics data using morphological analyzers.

'''


'''

|++++++++++++++++++++++++++|
| tokenize_ja(lyrics_file) |
|++++++++++++++++++++++++++|

tokenizes J-pop lyrics data by extracting nouns, verbs, and adjectives.

'''

def tokenize_ja(lyrics_file_ja):
    print("\n-------- J-POP LYRICS --------")

    # lyrics_file_ja = ../crawl_data/lyrics_jp/jp_lyrics_verbose.csv
    df = pd.read_csv(lyrics_file_ja)
    print(df.shape, "# as_is_jpop")
    df = df.dropna()
    print(df.shape, "# dropna()")
    df = df.drop_duplicates()
    print(df.shape, "# drop_duplicates()")

    data = list(df['Lyrics'].values)
    print("num_lyrics_jpop:", len(data))

    # Load Japanese stopwords.

    with open('stopwords/stopwords-ja.txt', 'r') as f:
        stopwords = f.read()
        stopwords = stopwords.split("\n")

    # Load Japanese morphological analyzer along with a special dictionary.

    NEOLOGD = "-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd"
    m = MeCab.Tagger(NEOLOGD)

    lines = []
    for lyric in data:
        # Remove English words.
        lyric = re.sub('[a-zA-z]', '', lyric)
        morph = []
        m.parse('')
        lex = m.parseToNode(re.sub('\u3000', ' ', lyric))
        while lex:
            # Insert tokens to dictionary
            tmp = {}
            tmp['surface'] = lex.surface
            tmp['base'] = lex.feature.split(',')[-3]  # base
            tmp['pos'] = lex.feature.split(',')[0]  # pos
            tmp['pos1'] = lex.feature.split(',')[1]  # pos1
            # Begin/end sentence markers are not included.
            if 'BOS/EOS' not in tmp['pos']:
                morph.append(tmp)
            lex = lex.next
        lines.append(morph)

    # If 'base' word exists, use 'base' word; otherwise use 'surface' word as dictionary index word.
    word_list = []
    for line in lines:
        tmp = []
        for morph in line:
            if (morph['pos'] == '名詞') | (morph['pos'] == '動詞') | (morph['pos'] == '形容詞'):
                if (not morph['base'] == '*') & (morph['base'] not in stopwords):
                    tmp.append("{}:{}".format(morph['base'], morph['pos']))
                elif (morph['surface'] not in stopwords):
                    tmp.append("{}:{}".format(morph['surface'], morph['pos']))
        word_list.append(tmp)

    # Save tokenized lyrics, which contains nouns, verbs, and adjectives, to a pickle file.

    with open("../data/tokeninzed_jpop.p", 'wb') as f:
        pickle.dump(word_list, f)

    print("word list jpop sample:", word_list[0])

    flat_list = [item for sublist in word_list for item in sublist]
    print("total_jpop_words:", len(flat_list))

    counts = Counter(flat_list)
    print("uniq_words_jpop:", len(counts))

    # Save unique word list with frequency to file.

    with open("../data/uniq_words_freq_jpop.txt", 'w') as f:
        for k, v in counts.most_common():
            f.write("{}\t{}\n".format(k, v))


'''

|++++++++++++++++++++++++++|
| tokenize_ko(lyrics_file) |
|++++++++++++++++++++++++++|

tokenizes K-pop lyrics data by extracting 
nouns (common noun & proper noun), verbs and adjectives.

'''

def tokenize_ko(lyrics_file_ko):
    print("\n-------- K-POP LYRICS --------")

    # lyrics_file_ko = "../crawl_data/lyrics_kr/kr_lyrics_verbose.csv"
    df = pd.read_csv(lyrics_file_ko)
    print(df.shape, "# as_is_kpop")
    df = df.dropna()
    print(df.shape, "# dropna()")
    df = df.drop_duplicates()
    print(df.shape, "# drop_duplicates()")

    data = list(df['Lyrics'].values)
    print("num_lyrics_kpop:", len(data))

    # Load Korean stopwords.

    stopwords = ["하:VV", "있:VV", "되:VV", "있:VA", "이러:VV"]

    # Load Korean morphological analyzer.

    mecab = Mecab()

    word_list = []
    for lyric in data:
        lyric = re.sub('[a-zA-z]', '', lyric)
        parsed = mecab.pos(lyric)
        tmp = []
        for w, pos in parsed:
            # We look for four parts of speech.
            # See below URL for POS tags (Mecab-ko).
            # *** KoNLPy Korean POS Tag Comparison Chart ***
            # https://docs.google.com/spreadsheets/d/1OGAjUvalBuX-oZvZ_-9tEfYD2gQe7hTGsgUpiiBSXI8/edit#gid=0
            if (pos == 'NNG') | (pos == 'NNP') | (pos == 'VV') | (pos == 'VA'):
                wpos = "{}:{}".format(w, pos)
                if wpos not in stopwords:
                    tmp.append(wpos)
        word_list.append(tmp)

    # Save tokenized lyrics, which contains nouns, verbs, and adjectives, to a pickle file.

    with open("../data/tokeninzed_kpop.p", 'wb') as f:
        pickle.dump(word_list, f)

    print("word list kpop sample:", word_list[0])

    flat_list = [item for sublist in word_list for item in sublist]
    print("total_kpop_words:", len(flat_list))

    counts = Counter(flat_list)
    print("uniq_words_kpop:", len(counts))

    # Save unique word list with frequency to file.

    with open("../data/uniq_words_freq_kpop.txt", 'w') as f:
        for k, v in counts.most_common():
            f.write("{}\t{}\n".format(k, v))


#---------------------------------------
# Tokenize j-pop lyrics data and extract nouns, verbs, and adjectives.

tokenize_ja("../data/lyrics_jp.csv")

#---------------------------------------
# Tokenize k-pop lyrics data and extract nouns (common nouns and proper nouns), verbs, and adjectives.

tokenize_ko("../data/lyrics_kr.csv")




'''

/usr/bin/python3 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/collabtech2019/find_latent_words/1_tokenize.py

-------- J-POP LYRICS --------
(1142, 6) # as_is_jpop
(1142, 6) # dropna()
(1142, 6) # drop_duplicates()
num_lyrics_jpop: 1142
word list jpop sample: ['ゆれる:動詞', '光:名詞', 'ひとつ:名詞', '痛む:動詞', '癒す:動詞', '消える:動詞', '落ちる:動詞', '涙。:名詞', 'ひとつ:名詞', '思い:名詞', '届く:動詞', '消える:動詞', '止まる:動詞', '時:名詞', '潜む:動詞', '愛:名詞', '降り注ぐ:動詞', '雨:名詞', 'こぼれ落ちる:動詞', '涙のあと:名詞', '凍える:動詞', 'そう:名詞', '涙の色:名詞', '戻れる:動詞', '記憶:名詞', '巡る:動詞', '全て:名詞', '奪う:動詞', 'この世の果て:名詞', '悲しみ:名詞', '終わる:動詞', '描く:動詞', '心:名詞', '謎:名詞', 'めく:動詞', '闇:名詞', '迫る:動詞', '真実:名詞', '世界:名詞', '描く:動詞', '明日:名詞', '見える:動詞', '百合:名詞', '汚れ:名詞', '知る:動詞', '願い:名詞', '透明:名詞', 'まま:名詞', '白い:形容詞', '染まる:動詞', '花:名詞', '変わる:動詞', '誓う:動詞', '届く:動詞', '声:名詞', '残る:動詞', '愛:名詞', '吹く:動詞', 'ぬける:動詞', '風:名詞', 'こぼれ落ちる:動詞', '涙のあと:名詞', '隠す:動詞', 'きれる:動詞', 'ふたつ:名詞', '顔:名詞', '終わる:動詞', '夜:名詞', '眠る:動詞', '夢:名詞', '傷跡:名詞', '残す:動詞', '痛み:名詞', '悲しみ:名詞', '僅か:名詞', '光:名詞', '生まれる:動詞', '嘆き:名詞', '繰り返す:動詞', '嘘:名詞', '消える:動詞', '真実:名詞', '最後:名詞', '羽:名詞', '開く:動詞', '運命:名詞', '定め:名詞', '変える:動詞', '百合:名詞', '花:名詞', '儚い:形容詞', 'げ:名詞', '痛み:名詞', '消える:動詞', '夢なら:名詞', '愛す:動詞', 'まま:名詞', '悲しみ:名詞', '終わる:動詞', '描く:動詞', '心:名詞', '謎:名詞', 'めく:動詞', '闇:名詞', '迫る:動詞', '真実:名詞', '世界:名詞', '描く:動詞', '明日:名詞', '見える:動詞', '百合:名詞', '汚れ:名詞', '知る:動詞', '願い:名詞', '透明:名詞', 'まま:名詞']
total_jpop_words: 125205
uniq_words_jpop: 13086

-------- K-POP LYRICS --------
(1000, 6) # as_is_kpop
(1000, 6) # dropna()
(1000, 6) # drop_duplicates()
num_lyrics_kpop: 1000
word list kpop sample: ['쳐다보:VV', '예쁘:VA', '그렇:VA', '쳐다보:VV', '쑥스럽:VA', '때:NNG', '마다:NNG', '돌리:VV', '남자:NNG', '뒤:NNG', '시선:NNG', '좋:VA', '매력:NNG', '눈길:NNG', '따라오:VV', '남자:NNG', '때:NNG', '같:VA', '부담:NNG', '살:VV', '여자:NNG', '애:NNG', '엄마:NNG', '날:NNG', '낳:VV', '삶:NNG', '피곤:NNG', '매력:NNG', '매력:NNG', '날:NNG', '다니:VV', '스포트:NNP', '라이트:NNP', '가:VV', '쫓아오:VV', '식당:NNG', '길거리:NNG', '카페:NNG', '나이트:NNG', '인기:NNG', '사그러들:VV', '원:NNG', '눈:NNG', '고소영:NNP', '하지원:NNP', '좋:VA', '좋:VA', '같:VA', '매력:NNG']
total_kpop_words: 77092
uniq_words_kpop: 5797

Process finished with exit code 0

'''

