import pandas as pd

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code analyzes the statistics of the J-pop/K-pop alignment dictionary. 

'''

'''

|++++++++++++++++++++|
| check_dictionary() |
|++++++++++++++++++++|

checks the content of the J-pop/K-pop lyrics word alignment dictionary.

'''

def check_dictionary():
    print("\n--------------------------------------------------------")
    print("      J-POP/K-POP LYRICS WORD ALIGNMENT DICTIONARY ")
    print("--------------------------------------------------------\n")
    df_dict = pd.read_csv('../dict/jpop_kpop_align_dict.csv', delimiter='\t')
    print(df_dict.head(3))
    print("\njako_dict shape:", df_dict.shape)

    # Counts individual dictionary words (ungrouped).

    indiv_dword_kpop = []
    indiv_dword_jpop = []
    for idx, row in df_dict.iterrows():
        ja_w = row[0]
        ko_w = row[1]
        if (':NNG' in ko_w) | (':NNP' in ko_w) | (':VV' in ko_w) | (':VA' in ko_w):
            if '|' in ko_w:
                indiv_dword_kpop += ko_w.split("|")
            else:
                indiv_dword_kpop.append(ko_w)
            if '|' in ja_w:
                indiv_dword_jpop += ja_w.split("|")
            else:
                indiv_dword_jpop.append(ja_w)
        else:
            print("other_POS:", ko_w)

    print("\n# of indiv. jpop_words in dict:", len(indiv_dword_jpop), ", check uniqueness:", len(set(indiv_dword_jpop)))
    print("# of indiv. kpop_words in dict:", len(indiv_dword_kpop), ", check uniqueness:", len(set(indiv_dword_kpop)))

    uniq_word_jpop = pd.read_csv("../data/uniq_words_freq_jpop.txt", delimiter='\t', header=None)
    uniq_word_kpop = pd.read_csv("../data/uniq_words_freq_kpop.txt", delimiter='\t', header=None)

    uw_jpop = uniq_word_jpop[0].values
    uw_kpop = uniq_word_kpop[0].values

    diff_jpop = set(indiv_dword_jpop) - set(uw_jpop)
    print("\ncheck diff_jpop:", len(diff_jpop), diff_jpop)

    diff_kpop = set(indiv_dword_kpop) - set(uw_kpop)
    print("check diff_kpop:", len(diff_kpop), diff_kpop)

    if '' in indiv_dword_jpop:
        print("empty string!")

    if '' in indiv_dword_kpop:
        print("empty string!")

    for ind, elem in enumerate(indiv_dword_jpop):
        if elem == '':
            print(ind)
            break

    for ind, elem in enumerate(indiv_dword_kpop):
        if elem == '':
            print(ind)
            break


#---------------------------------------
# Check the content of the manually created
# J-pop/K-pop lyrics word alignment dictionary.

check_dictionary()


'''
/usr/bin/python3 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/collabtech2019/find_latent_words/2_check_dict.py

--------------------------------------------------------
      J-POP/K-POP LYRICS WORD ALIGNMENT DICTIONARY 
--------------------------------------------------------

                                         J-POP   K-POP
0                                       あの日:名詞  그날:NNG
1                                     ありふれる:動詞   흔하:VA
2  いい:形容詞|よい:形容詞|良い:形容詞|イイ:形容詞|素晴らしい:形容詞|いい:動詞    좋:VA

jako_dict shape: (579, 2)

# of indiv. jpop_words in dict: 1065 , check uniqueness: 1065
# of indiv. kpop_words in dict: 870 , check uniqueness: 870

check diff_jpop: 0 set()
check diff_kpop: 0 set()

Process finished with exit code 0

'''