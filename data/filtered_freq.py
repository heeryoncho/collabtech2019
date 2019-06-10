import pickle
import collections
from itertools import chain

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code sorts the filtered lyrics words according to their frequency. 

'''

def sort_frequency():
    # =========================================================================

    print("\n*******************")
    print("       J-POP       ")
    print("*******************")

    with open("../data/filtered_jpop.p", 'rb') as f:
        jpop = pickle.load(f)
    flat_jpop = list(chain.from_iterable(jpop))
    freq_jpop = collections.Counter(flat_jpop)
    #print(freq_jpop)

    jpop_noun = [k for k, v in freq_jpop.items() if ':名詞' in k]
    dict_jpop_noun = dict()
    for w in jpop_noun:
        dict_jpop_noun[w] = freq_jpop[w]

    sorted_jpop_noun = sorted(dict_jpop_noun.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_jpop_noun.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_jpop_noun))

    jpop_verb = [k for k, v in freq_jpop.items() if ':動詞' in k]
    dict_jpop_verb = dict()
    for w in jpop_verb:
        dict_jpop_verb[w] = freq_jpop[w]

    sorted_jpop_verb = sorted(dict_jpop_verb.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_jpop_verb.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_jpop_verb))

    jpop_adj = [k for k, v in freq_jpop.items() if ':形容詞' in k]
    dict_jpop_adj = dict()
    for w in jpop_adj:
        dict_jpop_adj[w] = freq_jpop[w]

    sorted_jpop_adj = sorted(dict_jpop_adj.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_jpop_adj.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_jpop_adj))

    # =========================================================================

    print("\n*******************")
    print("       K-POP       ")
    print("*******************")

    with open("../data/filtered_kpop.p", 'rb') as f:
        kpop = pickle.load(f)
    flat_kpop = list(chain.from_iterable(kpop))
    freq_kpop = collections.Counter(flat_kpop)
    #print(freq_kpop)

    kpop_noun = [k for k, v in freq_kpop.items() if ':名詞' in k]
    dict_kpop_noun = dict()
    for w in kpop_noun:
        dict_kpop_noun[w] = freq_kpop[w]

    sorted_kpop_noun = sorted(dict_kpop_noun.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_kpop_noun.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_kpop_noun))

    kpop_verb = [k for k, v in freq_kpop.items() if ':動詞' in k]
    dict_kpop_verb = dict()
    for w in kpop_verb:
        dict_kpop_verb[w] = freq_kpop[w]

    sorted_kpop_verb = sorted(dict_kpop_verb.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_kpop_verb.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_kpop_verb))

    kpop_adj = [k for k, v in freq_kpop.items() if ':形容詞' in k]
    dict_kpop_adj = dict()
    for w in kpop_adj:
        dict_kpop_adj[w] = freq_kpop[w]

    sorted_kpop_adj = sorted(dict_kpop_adj.items(), key=lambda x: x[1], reverse=True)

    with open("topn_freq_kpop_adj.txt", 'w') as f:
        f.write('\n'.join('{} {}'.format(x[0], x[1]) for x in sorted_kpop_adj))

sort_frequency()