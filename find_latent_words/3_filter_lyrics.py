import os
import pickle

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code filters the lyrics words using J-pop/K-pop alignment dictionary.

Also, K-pop lyrics words are converted to Japanese using the dictionary.
This is done to unify the two languages into one (i.e., to Japanese).

Mapping and filtering are performed using J-pop/K-pop lyrics word alignment dictionary.

   <sample entry in the J-pop/K-pop lyrics word alignment dictionary>
   夢:名詞|ゆめいっぱい:名詞          꿈:NNG

   J-pop lyrics word '夢:名詞' is replaced with '夢:名詞|ゆめいっぱい:名詞' through filtering.
   K-pop lyrics word '꿈:NNG' is converted to '夢:名詞|ゆめいっぱい:名詞' through mapping.

Hence, through mapping and filtering process, both J-pop and K-pop lyrics words are 
selected and converted to Japanese alignment dictionary words.

'''


'''
|+++++++++++++++++|
| filter_lyrics() |
|+++++++++++++++++|

filters the lyrics data using the following 3 files:

1. '../dict/jpop_kpop_align_dict.p'
2. '../data/tokeninzed_jpop.p' file
3. '../data/tokeninzed_kpop.p' file

the output files are:

1. "../data/filtered_jpop.p"
2. "../data/filtered_kpop.p"

the filtered lyrics data, both J-pop and K-pop, 
contain "Japanese" index words listed in the 
J-pop/K-pop lyrics word alignment dictionary. 

'''

def filter_lyrics():
    print("\n-------- FILTERED LYRICS --------")

    # Load J-pop/K-pop lyrics word alignment dictionary.

    jako_dict = pickle.load(open("../dict/jpop_kpop_align_dict.p", 'rb'))

    # Reverse key and value to create koja_dict.

    koja_dict = {v: k for k, v in jako_dict.items()}

    # Split grouped Japanese words.

    jpop_list = list(koja_dict.values())
    jpop_hash = {}
    for ja in jpop_list:
        if "|" in ja:
            splitted_ja = ja.split("|")
            for each in splitted_ja:
                jpop_hash[each] = ja
        else:
            jpop_hash[ja] = ja

    #======================================================
    # Read J-pop lyrics data.

    with open("../data/tokeninzed_jpop.p", 'rb') as f:
        jpop_lyrics = pickle.load(f)

    jpop_filtered = []
    for each_lyric in jpop_lyrics:
        tmp = []
        for w in each_lyric:
            if w in jpop_hash:
                tmp.append(jpop_hash[w])
        if tmp != []:
            jpop_filtered.append(tmp)
    print("\n# of matching J-pop lyrics texts:", len(jpop_filtered))

    flat_list_jpop = [item for sublist in jpop_filtered for item in sublist]
    print("total num of filtered J-pop words:", len(flat_list_jpop))

    # Save J-pop_filtered lyrics.

    with open("../data/filtered_jpop.p", 'wb') as f:
        pickle.dump(jpop_filtered, f)

    #======================================================
    # Read K-pop lyrics data.

    with open("../data/tokeninzed_kpop.p", 'rb') as f:
        kpop_lyrics = pickle.load(f)

    kpop_list = list(koja_dict.keys())
    kpop_hash = {}
    for ko in kpop_list:
        if "|" in ko:
            splitted_ko = ko.split("|")
            for each in splitted_ko:
                kpop_hash[each] = ko
        else:
            kpop_hash[ko] = ko

    kpop_filtered = []
    for each_lyric in kpop_lyrics:
        tmp = []
        for w in each_lyric:
            if w in kpop_hash:
                tmp.append(koja_dict[kpop_hash[w]])
        if tmp != []:
            kpop_filtered.append(tmp)
    print("\n# of matching K-pop lyrics texts:", len(kpop_filtered))

    flat_list_kpop = [item for sublist in kpop_filtered for item in sublist]
    print("total num of filtered K-pop words:", len(flat_list_kpop))

    # Save K-pop_filtered lyrics.
    # *** Note that Korean lyrics words are mapped to Japanese dictionary words. ***

    with open("../data/filtered_kpop.p", 'wb') as f:
        pickle.dump(kpop_filtered, f)

    # Merge filtered_ja & filtered_ko lyrics
    filtered = jpop_filtered + kpop_filtered
    print("\ntotal num of lyrics (jpop+kpop):", len(filtered))


#---------------------------------------
# Filter k-pop & j-pop lyrics data using the alignment dictionary and
# save the result to 'filtered_lyrics' directory.
# The filtered data are used hereinafter.

filter_lyrics()




'''

/usr/bin/python3 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/collabtech2019/find_latent_words/1_tokenize.py

-------- FILTERED LYRICS --------

# of matching J-pop lyrics texts: 1134
total num of filtered J-pop words: 61094

# of matching K-pop lyrics texts: 986
total num of filtered K-pop words: 57049

total num of lyrics (jpop+kpop): 2120

Process finished with exit code 0

'''