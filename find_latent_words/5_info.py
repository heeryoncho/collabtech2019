import pickle
from itertools import chain

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code analyzes the statics of the lyrics data and the coverage of the dictionary words.

'''

'''

|+++++++++++++++++++|
| data_statistics() |
|+++++++++++++++++++|

calculates the statics of the lyrics data and the coverage of the 
J-pop/K-pop alignment dictionary words. 

'''


def data_statistics():
    with open("../data/tokeninzed_jpop.p", 'rb') as f:
        raw_lyrics_jpop = pickle.load(f)
    print("\nall lyrics (J-POP):", len(raw_lyrics_jpop))

    with open("../data/tokeninzed_kpop.p", 'rb') as f:
        raw_lyrics_kpop = pickle.load(f)
    print("all lyrics (K-POP):", len(raw_lyrics_kpop))

    print("\nsample lyrics (J-POP):", raw_lyrics_jpop[0][:5])
    print("sample lyrics (K-POP):", raw_lyrics_kpop[0][:5])

    flat_jpop = list(chain.from_iterable(raw_lyrics_jpop))
    flat_kpop = list(chain.from_iterable(raw_lyrics_kpop))

    print("\ntotal words (J-POP):", len(flat_jpop))
    print("total words (K-POP):", len(flat_kpop))

    print("\ntotal uniq. words (J-POP):", len(set(flat_jpop)))
    print("total uniq. words (K-POP):", len(set(flat_kpop)))

    print("\navg. words per lyric (J-POP):",
          round(float(len(flat_jpop)) / len(raw_lyrics_jpop), 2))
    print("avg. words per lyric (K-POP):",
          round(float(len(flat_kpop)) / len(raw_lyrics_kpop), 2))

    jpop_noun = []
    jpop_verb = []
    jpop_adj = []
    for lyric in raw_lyrics_jpop:
        for word in lyric:
            if ":名詞" in word:
                jpop_noun.append(word)
            if ":動詞" in word:
                jpop_verb.append(word)
            if ":形容詞" in word:
                jpop_adj.append(word)

    print("\n# nouns (J-POP):", len(jpop_noun))
    print("# verbs (J-POP):", len(jpop_verb))
    print("# adjectives (J-POP):", len(jpop_adj))

    # Check total number of J-pop words.
    sum_3pos_jpop = len(jpop_noun) + len(jpop_verb) + len(jpop_adj)
    if sum_3pos_jpop == len(flat_jpop):
        print("\n--- total jpop words checks out:", sum_3pos_jpop, len(flat_jpop))
    else:
        print("\n--- SOMETHING'S WRONG with J-pop word total.")

    print("\n# uniq nouns (J-POP):", len(set(jpop_noun)))
    print("# uniq verbs (J-POP):", len(set(jpop_verb)))
    print("# uniq adjectives (J-POP):", len(set(jpop_adj)))

    # Check total number of unique J-pop words.
    sum_3pos_jpop_uniq = len(set(jpop_noun)) + len(set(jpop_verb)) + len(set(jpop_adj))
    if sum_3pos_jpop_uniq == len(set(flat_jpop)):
        print("\n--- total unique kpop words checks out:", sum_3pos_jpop_uniq, len(set(flat_jpop)))
    else:
        print("\n--- SOMETHING'S WRONG with unique K-pop word total.")

    #=========================================

    kpop_noun = []
    kpop_verb = []
    kpop_adj = []
    for lyric in raw_lyrics_kpop:
        for word in lyric:
            if ":NNG" in word or ":NNP" in word:
                kpop_noun.append(word)
            if ":VV" in word:
                kpop_verb.append(word)
            if ":VA" in word:
                kpop_adj.append(word)

    print("\n# nouns (K-POP):", len(kpop_noun))
    print("# verbs (K-POP):", len(kpop_verb))
    print("# adjectives (K-POP):", len(kpop_adj))

    # Check total number of K-pop words.
    sum_3pos_kpop = len(kpop_noun) + len(kpop_verb) + len(kpop_adj)
    if sum_3pos_kpop == len(flat_kpop):
        print("\n--- total kpop words checks out:", sum_3pos_kpop, len(flat_kpop))
    else:
        print("\n--- SOMETHING'S WRONG with K-pop word total.")

    print("\n# uniq nouns (K-POP):", len(set(kpop_noun)))
    print("# uniq verbs (K-POP):", len(set(kpop_verb)))
    print("# uniq adjectives (K-POP):", len(set(kpop_adj)))

    # Check total number of unique K-pop words.
    sum_3pos_kpop_uniq = len(set(kpop_noun)) + len(set(kpop_verb)) + len(set(kpop_adj))
    if sum_3pos_kpop_uniq == len(set(flat_kpop)):
        print("\n--- total unique kpop words checks out:", sum_3pos_kpop_uniq, len(set(flat_kpop)))
    else:
        print("\n--- SOMETHING'S WRONG with unique K-pop word total.")

    print("\n===============================================")

    with open("../data/filtered_jpop.p", 'rb') as f:
        lyrics_jpop = pickle.load(f)
    print("\nall lyrics (FILTERED J-POP):", len(lyrics_jpop))

    with open("../data/filtered_kpop.p", 'rb') as f:
        lyrics_kpop = pickle.load(f)
    print("all lyrics (FILTERED K-POP):", len(lyrics_kpop))

    print("\nsample lyrics (FILTERED J-POP):", lyrics_jpop[0][:5])
    print("sample lyrics (FILTERED K-POP):", lyrics_kpop[0][:5])

    filtered_jpop = list(chain.from_iterable(lyrics_jpop))
    filtered_kpop = list(chain.from_iterable(lyrics_kpop))

    print("\ntotal words (FILTERED J-POP):", len(filtered_jpop))
    print("total words (FILTERED K-POP):", len(filtered_kpop))

    print("\ntotal uniq. words (FILTERED J-POP):", len(set(filtered_jpop)))
    print("total uniq. words (FILTERED K-POP):", len(set(filtered_kpop)))

    print("\navg. words per lyric (FILTERED J-POP):",
          round(float(len(filtered_jpop)) / len(lyrics_jpop), 2))
    print("avg. words per lyric (FILTERED K-POP):",
          round(float(len(filtered_kpop)) / len(lyrics_kpop), 2))

    print("\ndictionary coverage (J-POP):",
          round(float(len(filtered_jpop)) / len(flat_jpop), 4))
    print("dictionary coverage (K-POP):",
          round(float(len(filtered_kpop)) / len(flat_kpop), 4))



# Analyzes the statistics of the lyrics data.

data_statistics()


'''

/usr/bin/python3 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/collabtech2019/find_latent_words/5_info.py

all lyrics (J-POP): 1142
all lyrics (K-POP): 1000

sample lyrics (J-POP): ['ゆれる:動詞', '光:名詞', 'ひとつ:名詞', '痛む:動詞', '癒す:動詞']
sample lyrics (K-POP): ['쳐다보:VV', '예쁘:VA', '그렇:VA', '쳐다보:VV', '쑥스럽:VA']

total words (J-POP): 125205
total words (K-POP): 77092

total uniq. words (J-POP): 13086
total uniq. words (K-POP): 5797

avg. words per lyric (J-POP): 109.64
avg. words per lyric (K-POP): 77.09

# nouns (J-POP): 77859
# verbs (J-POP): 41353
# adjectives (J-POP): 5993

--- total jpop words checks out: 125205 125205

# uniq nouns (J-POP): 9876
# uniq verbs (J-POP): 2799
# uniq adjectives (J-POP): 411

--- total unique kpop words checks out: 13086 13086

# nouns (K-POP): 51161
# verbs (K-POP): 18413
# adjectives (K-POP): 7518

--- total kpop words checks out: 77092 77092

# uniq nouns (K-POP): 4478
# uniq verbs (K-POP): 1063
# uniq adjectives (K-POP): 256

--- total unique kpop words checks out: 5797 5797

===============================================

all lyrics (FILTERED J-POP): 1134
all lyrics (FILTERED K-POP): 986

sample lyrics (FILTERED J-POP): ['光:名詞', '痛い:形容詞|病む:動詞|病める:動詞|いたい:形容詞|痛む:動詞|痛み:名詞|痛:名詞', '消える:動詞|消す:動詞', '落ちる:動詞|流れ落ちる:動詞', '涙:名詞|涙。:名詞|涙-NAMIDA-:名詞']
sample lyrics (FILTERED K-POP): ['見る:動詞|みる:動詞|見つめる:動詞|見上げる:動詞|眺める:動詞|見える:動詞|みせる:動詞', 'かわいい:形容詞|綺麗:名詞|きれい:名詞', '見る:動詞|みる:動詞|見つめる:動詞|見上げる:動詞|眺める:動詞|見える:動詞|みせる:動詞', '時:名詞|頃:名詞', '回す:動詞']

total words (FILTERED J-POP): 61094
total words (FILTERED K-POP): 57049

total uniq. words (FILTERED J-POP): 579
total uniq. words (FILTERED K-POP): 579

avg. words per lyric (FILTERED J-POP): 53.87
avg. words per lyric (FILTERED K-POP): 57.86

dictionary coverage (J-POP): 0.488
dictionary coverage (K-POP): 0.74

Process finished with exit code 0

'''