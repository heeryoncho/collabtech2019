import os
import pickle
import numpy as np
import pandas as pd
import gensim
from gensim.models.keyedvectors import KeyedVectors as kv
from find_latent_words import tensorly_modified
from functools import reduce

'''

# Author: Heeryon Cho <heeryon.cho@gmail.com>
# License: BSD-3-clause

This code generates J-pop & K-pop word2vec models using the filtered J-pop/K-pop lyrics:

--- '../data/filtered_jpop.p' file
--- '../data/filtered_kpop.p' file

This code outputs varying J-pop/Both/K-pop word2vec models:
(one seed & 3 different word embedding dimensions [5, 10, 100])

--- 'word2vec/w2v_both_s{}_d{}.kv'
--- 'word2vec/w2v_jpop_s{}_d{}.kv'
--- 'word2vec/w2v_kpop_s{}_d{}.kv'

This code outputs sorted J-pop/Both/K-pop CPD word lists:
(This code uses three different random state values to
 output three different CP decomposition results for J-pop/Both/K-pop.)

--- 'cpd_result/both_s0_d5_c{}.txt'
--- 'cpd_result/jpop_s0_d5_c{}.txt'
--- 'cpd_result/kpop_s0_d5_c{}.txt'

Then the three random state value results are merged for each J-pop/Both/K-pop:

--- 'cpd_result/both_final.txt'
--- 'cpd_result/jpop_final.txt'
--- 'cpd_result/kpop_final.txt'

'''

'''

|++++++++++++++++++++++++|
| load_filtered_lyrics() |
|++++++++++++++++++++++++|

load filtered J-pop & K-pop lyrics data.

'''


def load_filtered_lyrics():
    with open("../data/filtered_jpop.p", 'rb') as f:
        lyrics_jpop = pickle.load(f)
    # print("# of jpop lyrics text:", len(lyrics_jpop))

    with open("../data/filtered_kpop.p", 'rb') as f:
        lyrics_kpop = pickle.load(f)
    # print("# of kpop lyrics text:", len(lyrics_kpop))

    return lyrics_jpop, lyrics_kpop


# Define new hash function for reproducibility.
# This function is only used in the word2vec() function below.

def new_hash(selected):
    return ord(selected[0])


'''

|++++++++++++|
| word2vec() |
|++++++++++++|

builds and saves to file three word2vec models, 
one for J-pop, one for K-pop, and one for Both (J-pop & K-pop).

'''


def word2vec(w2v_seed=0, w2v_dim=0):
    # Create 'word2vec' directory if there isn't any.

    word2vec_dir = "word2vec"
    if not os.path.exists(word2vec_dir):
        os.makedirs(word2vec_dir)

    # Load filtered lyrics data.

    lyrics_jpop, lyrics_kpop = load_filtered_lyrics()

    # Merge lyrics_jpop & lyrics_kpop lyrics.

    lyrics = lyrics_jpop + lyrics_kpop

    # ------------------------------------------------------------------
    # Build J-pop word2vec model.
    # ------------------------------------------------------------------
    # Update initial word2vec model using only lyrics_jpop data.

    model_jpop = gensim.models.Word2Vec(lyrics, size=w2v_dim, window=5, min_count=1,
                                        workers=1, seed=w2v_seed, hashfxn=new_hash)

    # Retrain word2vec model using only lyrics_jpop data to create J-pop word2vec model.

    model_jpop.train(lyrics_jpop, total_examples=len(lyrics_jpop), epochs=50)

    # Save word2vec model with '.kv' extension ('kv' stands for 'keyvector').

    model_jpop.wv.save("word2vec/w2v_jpop_s{}_d{}.kv".format(w2v_seed, w2v_dim))

    # ------------------------------------------------------------------
    # Build K-pop word2vec model.
    # ------------------------------------------------------------------
    # Update initial word2vec model using lyrics_kpop data.

    model_kpop = gensim.models.Word2Vec(lyrics, size=w2v_dim, window=5, min_count=1,
                                        workers=1, seed=w2v_seed, hashfxn=new_hash)

    # Retrain word2vec model using only lyrics_kpop data to create K-pop word2vec model.

    model_kpop.train(lyrics_kpop, total_examples=len(lyrics_kpop), epochs=50)

    # Save word2vec model with '.kv' extension ('kv' stands for 'keyvector').

    model_kpop.wv.save("word2vec/w2v_kpop_s{}_d{}.kv".format(w2v_seed, w2v_dim))

    # ------------------------------------------------------------------
    # Build (initial) word2vec model.
    # ------------------------------------------------------------------
    # Train word2vec model using both lyrics_jpop & lyrics_kpop data.

    model_both = gensim.models.Word2Vec(lyrics, size=w2v_dim, window=5, min_count=1,
                                        workers=1, seed=w2v_seed, hashfxn=new_hash)

    # Save word2vec model with '.kv' extension ('kv' stands for 'keyvector').

    model_both.wv.save("word2vec/w2v_both_s{}_d{}.kv".format(w2v_seed, w2v_dim))


word2vec(w2v_seed=0, w2v_dim=5)

'''

|++++++++++++++++|
| CPD_wordlist() |
|++++++++++++++++|

outputs rank-3 mode-3 CP decomposition tensor using fixed mode-3 values by
importing the 'tensorly_modified' code that modifies the 'tensorly' library.

this modified code performs a partially fixed CP decomposition.

from the decomposition results, mode-1 vector values are sorted
in a descending order and the corresponding index words are laid out.

the sorted index words are saved to file in the 'cpd_result' directory.

'''


def gen_CPD_wordlist(verbose=True, w2v_seed=0, w2v_dim=0, rs=0, country_values=[]):
    # Create 'cpd_result' directory if there isn't any.

    cpd_dir = "cpd_result"
    if not os.path.exists(cpd_dir):
        os.makedirs(cpd_dir)

    kv_jpop = kv.load('word2vec/w2v_jpop_s{}_d{}.kv'.format(w2v_seed, w2v_dim))
    kv_both = kv.load('word2vec/w2v_both_s{}_d{}.kv'.format(w2v_seed, w2v_dim))
    kv_kpop = kv.load('word2vec/w2v_kpop_s{}_d{}.kv'.format(w2v_seed, w2v_dim))

    print("kv_jpop.vectors.shape:", kv_jpop.vectors.shape)

    X = np.stack((kv_jpop.vectors, kv_both.vectors, kv_kpop.vectors), axis=2)
    print("stacked X shape:", X.shape)

    # fixed_jpop = [0.8, 0.1, 0.1]
    # fixed_both = [0.1, 0.8, 0.1]
    # fixed_kpop = [0.1, 0.1, 0.8]

    # country_values = [fixed_jpop, fixed_both, fixed_kpop]

    # It is important to fix the random_state in order to obtain consistent results.
    # Here, consistent results mean generally consistent results of mode-1 word ordering.

    # To ensure convergence, n_iter_max is set at 1000.

    cpd = tensorly_modified.parafac(X, 3, random_state=rs, n_iter_max=1000,
                                    mode_three_val=country_values, verbose=verbose)

    # print(type(cpd))   # list
    print("Mode A shape:", cpd[0].shape)
    print("Mode B shape:", cpd[1].shape)
    print("Mode C shape:", cpd[2].shape)

    r_a = np.linalg.matrix_rank(cpd[0])
    print("Rank A:", r_a)
    r_b = np.linalg.matrix_rank(cpd[1])
    print("Rank B:", r_b)
    r_c = np.linalg.matrix_rank(cpd[2])
    print("Rank C:", r_c)

    corr_a = np.corrcoef(cpd[0].T)
    print("Corrcoef A:\n", corr_a)
    corr_b = np.corrcoef(cpd[1].T)
    print("Corrcoef B:\n", corr_b)
    corr_c = np.corrcoef(cpd[2].T)
    print("Corrcoef C:\n", corr_c)

    print("Mode C rank-1 vector values:\n", cpd[2])

    # Select mode-1 vectors containing values for the index words and transpose it.

    result = cpd[0].T

    dict_index = kv_jpop.index2word

    country = ["jpop", "both", "kpop"]

    output = []

    for c in range(3):  # c denotes either 'jpop', 'both', 'kpop'.
        val = {}
        for w in range(len(dict_index)):
            val[dict_index[w]] = result[c][w]
        output.append(val)

        sorted_list = [(k, val[k]) for k in sorted(val, key=val.get, reverse=True)]
        result_str = ""
        word_list = []
        for k, v in sorted_list:
            result_str += "{}\t{}\n".format(k, v)
            word_list.append(k)
        # print(result_str)
        result_file = "{}/{}_s{}_d{}_c{}.txt".format(cpd_dir, country[c], w2v_seed, w2v_dim, rs)
        with open(result_file, 'w') as f:
            f.write(result_str)

    return output


# Values of mode-3 vectors that represent country factor.

fixed_jpop = [0.8, 0.1, 0.1]
fixed_both = [0.1, 0.8, 0.1]
fixed_kpop = [0.1, 0.1, 0.8]

country_values = [fixed_jpop, fixed_both, fixed_kpop]

# w2v_seed is 0; w2v_dim is 5; cpd_rs is varied.

print("\n---------------------------------------------------------------------------")
print("    Condition: w2v_seed is fixed; w2v_dim is fixed; cpd_rs is [0, 1, 2]")
print("---------------------------------------------------------------------------\n")

for i in range(3):
    print("\n====== w2v_seed=0, w2v_dim=5, cpd_rs={} ======".format(i))
    gen_CPD_wordlist(w2v_seed=0, w2v_dim=5, rs=i, country_values=country_values)


'''

|+++++++++++++++++++++|
| merge_CPD_results() |
|+++++++++++++++++++++|

merges the three random initialization results for each of the
J-pop/Both/K-pop CPD results by taking the sum and sorting the CPD values. 

'''

def merge_CPD_results(jbk="jpop/both/kpop", top_bottom="t/b"):
   cpd_df = []
   for i in range(3):
       cpd_file = "cpd_result/{}_s0_d5_c{}.txt".format(jbk, i)
       df = pd.read_csv(cpd_file, header=None, sep='\t',
                        names=['word', 'value_{}'.format(i)])
       #print(df.shape)
       #print(df.head(3))
       #print(top_bottom[i])
       if top_bottom[i] == 'b':
           df['value_{}'.format(i)] = df['value_{}'.format(i)] * -1
       cpd_df.append(df)
   df_merged = reduce(lambda left, right: pd.merge(left, right, on=['word'], how='outer'), cpd_df)
   #print(df_merged.head(3))
   df_merged['sum'] = df_merged.iloc[:, 1:3].sum(axis=1)
   #print(df_merged.head(3))
   final_df = df_merged.sort_values(by='sum', ascending=False)
   #print(final_df.head(3))
   final_df.iloc[:, [0, 4]].to_csv("cpd_result/{}_final.txt".format(jbk),
                                   sep='\t', encoding='utf-8', index=False)


tb_jpop = ['b', 'b', 'b']
tb_both = ['t', 't', 't']
tb_kpop = ['b', 't', 't']

top_bottom = [tb_jpop, tb_both, tb_kpop]

jbk = ['jpop', 'both', 'kpop']

for c in range(3):
    merge_CPD_results(jbk=jbk[c], top_bottom=top_bottom[c])



'''
/usr/bin/python3 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/collabtech2019/find_latent_words/4_w2v_cpd.py

---------------------------------------------------------------------------
    Condition: w2v_seed is fixed; w2v_dim is fixed; cpd_rs is [0, 1, 2]
---------------------------------------------------------------------------

====== w2v_seed=0, w2v_dim=5, cpd_rs=0 ======
kv_jpop.vectors.shape: (579, 5)
stacked X shape: (579, 5, 3)
reconstruction error=0.7724369764328003, variation=0.0038328170776367188.
reconstruction error=0.7707643508911133, variation=0.0016726255416870117.
reconstruction error=0.7675790190696716, variation=0.0031853318214416504.
reconstruction error=0.7615362405776978, variation=0.006042778491973877.
reconstruction error=0.752296507358551, variation=0.009239733219146729.
reconstruction error=0.7417590618133545, variation=0.010537445545196533.
reconstruction error=0.7330713868141174, variation=0.00868767499923706.
reconstruction error=0.7276188135147095, variation=0.005452573299407959.
reconstruction error=0.7247447967529297, variation=0.002874016761779785.
reconstruction error=0.7233511805534363, variation=0.0013936161994934082.
reconstruction error=0.7226862907409668, variation=0.0006648898124694824.
reconstruction error=0.7223570346832275, variation=0.0003292560577392578.
reconstruction error=0.7221784591674805, variation=0.0001785755157470703.
reconstruction error=0.7220672369003296, variation=0.0001112222671508789.
reconstruction error=0.7219863533973694, variation=8.088350296020508e-05.
reconstruction error=0.721919596195221, variation=6.67572021484375e-05.
reconstruction error=0.7218599319458008, variation=5.9664249420166016e-05.
reconstruction error=0.7218042016029358, variation=5.5730342864990234e-05.
reconstruction error=0.7217512130737305, variation=5.2988529205322266e-05.
reconstruction error=0.7217005491256714, variation=5.066394805908203e-05.
reconstruction error=0.7216519713401794, variation=4.857778549194336e-05.
reconstruction error=0.7216055393218994, variation=4.64320182800293e-05.
reconstruction error=0.7215613126754761, variation=4.4226646423339844e-05.
reconstruction error=0.7215191721916199, variation=4.214048385620117e-05.
reconstruction error=0.7214791774749756, variation=3.999471664428711e-05.
reconstruction error=0.7214412093162537, variation=3.796815872192383e-05.
reconstruction error=0.7214052081108093, variation=3.600120544433594e-05.
reconstruction error=0.7213712930679321, variation=3.3915042877197266e-05.
reconstruction error=0.721339225769043, variation=3.2067298889160156e-05.
reconstruction error=0.7213088870048523, variation=3.0338764190673828e-05.
reconstruction error=0.7212803363800049, variation=2.855062484741211e-05.
reconstruction error=0.7212533950805664, variation=2.6941299438476562e-05.
reconstruction error=0.7212280035018921, variation=2.5391578674316406e-05.
reconstruction error=0.7212041020393372, variation=2.390146255493164e-05.
reconstruction error=0.7211815118789673, variation=2.2590160369873047e-05.
reconstruction error=0.7211602330207825, variation=2.1278858184814453e-05.
reconstruction error=0.7211402654647827, variation=1.996755599975586e-05.
reconstruction error=0.7211212515830994, variation=1.901388168334961e-05.
reconstruction error=0.7211035490036011, variation=1.7702579498291016e-05.
reconstruction error=0.7210867404937744, variation=1.6808509826660156e-05.
reconstruction error=0.7210709452629089, variation=1.5795230865478516e-05.
reconstruction error=0.7210560441017151, variation=1.4901161193847656e-05.
reconstruction error=0.7210419774055481, variation=1.4066696166992188e-05.
reconstruction error=0.7210286855697632, variation=1.329183578491211e-05.
reconstruction error=0.7210161685943604, variation=1.2516975402832031e-05.
reconstruction error=0.7210044264793396, variation=1.1742115020751953e-05.
reconstruction error=0.7209932208061218, variation=1.1205673217773438e-05.
reconstruction error=0.7209826707839966, variation=1.055002212524414e-05.
reconstruction error=0.7209727764129639, variation=9.894371032714844e-06.
reconstruction error=0.7209633588790894, variation=9.417533874511719e-06.
reconstruction error=0.7209545373916626, variation=8.821487426757812e-06.
reconstruction error=0.7209461331367493, variation=8.404254913330078e-06.
reconstruction error=0.7209382057189941, variation=7.927417755126953e-06.
reconstruction error=0.7209307551383972, variation=7.450580596923828e-06.
reconstruction error=0.720923662185669, variation=7.092952728271484e-06.
reconstruction error=0.7209169864654541, variation=6.67572021484375e-06.
reconstruction error=0.7209106683731079, variation=6.318092346191406e-06.
reconstruction error=0.7209047079086304, variation=5.9604644775390625e-06.
reconstruction error=0.7208990454673767, variation=5.662441253662109e-06.
reconstruction error=0.7208936810493469, variation=5.364418029785156e-06.
reconstruction error=0.720888614654541, variation=5.066394805908203e-06.
reconstruction error=0.720883846282959, variation=4.76837158203125e-06.
reconstruction error=0.720879316329956, variation=4.5299530029296875e-06.
reconstruction error=0.7208749651908875, variation=4.351139068603516e-06.
reconstruction error=0.7208709716796875, variation=3.993511199951172e-06.
reconstruction error=0.7208670377731323, variation=3.933906555175781e-06.
reconstruction error=0.7208634614944458, variation=3.5762786865234375e-06.
reconstruction error=0.720859944820404, variation=3.516674041748047e-06.
reconstruction error=0.7208566665649414, variation=3.2782554626464844e-06.
reconstruction error=0.7208536267280579, variation=3.039836883544922e-06.
reconstruction error=0.7208507061004639, variation=2.9206275939941406e-06.
reconstruction error=0.7208479046821594, variation=2.8014183044433594e-06.
reconstruction error=0.7208452820777893, variation=2.6226043701171875e-06.
reconstruction error=0.720842719078064, variation=2.562999725341797e-06.
reconstruction error=0.7208404541015625, variation=2.2649765014648438e-06.
reconstruction error=0.7208381295204163, variation=2.3245811462402344e-06.
reconstruction error=0.7208360433578491, variation=2.086162567138672e-06.
reconstruction error=0.7208340167999268, variation=2.0265579223632812e-06.
reconstruction error=0.720832109451294, variation=1.9073486328125e-06.
reconstruction error=0.7208303213119507, variation=1.7881393432617188e-06.
reconstruction error=0.7208285331726074, variation=1.7881393432617188e-06.
reconstruction error=0.7208269238471985, variation=1.6093254089355469e-06.
reconstruction error=0.7208253145217896, variation=1.6093254089355469e-06.
reconstruction error=0.7208238840103149, variation=1.430511474609375e-06.
reconstruction error=0.7208224534988403, variation=1.430511474609375e-06.
reconstruction error=0.7208212018013, variation=1.2516975402832031e-06.
reconstruction error=0.7208199501037598, variation=1.2516975402832031e-06.
reconstruction error=0.7208187580108643, variation=1.1920928955078125e-06.
reconstruction error=0.7208176255226135, variation=1.1324882507324219e-06.
reconstruction error=0.7208165526390076, variation=1.0728836059570312e-06.
reconstruction error=0.7208155393600464, variation=1.0132789611816406e-06.
reconstruction error=0.72081458568573, variation=9.5367431640625e-07.
reconstruction error=0.7208136320114136, variation=9.5367431640625e-07.
reconstruction error=0.7208128571510315, variation=7.748603820800781e-07.
reconstruction error=0.7208120226860046, variation=8.344650268554688e-07.
reconstruction error=0.7208111882209778, variation=8.344650268554688e-07.
reconstruction error=0.7208104133605957, variation=7.748603820800781e-07.
reconstruction error=0.7208097577095032, variation=6.556510925292969e-07.
reconstruction error=0.7208090424537659, variation=7.152557373046875e-07.
reconstruction error=0.7208083868026733, variation=6.556510925292969e-07.
reconstruction error=0.7208078503608704, variation=5.364418029785156e-07.
reconstruction error=0.7208072543144226, variation=5.960464477539062e-07.
reconstruction error=0.7208067774772644, variation=4.76837158203125e-07.
reconstruction error=0.7208061814308167, variation=5.960464477539062e-07.
reconstruction error=0.7208057641983032, variation=4.172325134277344e-07.
reconstruction error=0.7208052277565002, variation=5.364418029785156e-07.
reconstruction error=0.7208048701286316, variation=3.5762786865234375e-07.
reconstruction error=0.7208043932914734, variation=4.76837158203125e-07.
reconstruction error=0.72080397605896, variation=4.172325134277344e-07.
reconstruction error=0.7208036780357361, variation=2.980232238769531e-07.
reconstruction error=0.7208032608032227, variation=4.172325134277344e-07.
reconstruction error=0.720802903175354, variation=3.5762786865234375e-07.
reconstruction error=0.7208026051521301, variation=2.980232238769531e-07.
reconstruction error=0.7208023071289062, variation=2.980232238769531e-07.
reconstruction error=0.7208020091056824, variation=2.980232238769531e-07.
reconstruction error=0.7208017706871033, variation=2.384185791015625e-07.
reconstruction error=0.7208015322685242, variation=2.384185791015625e-07.
reconstruction error=0.7208012342453003, variation=2.980232238769531e-07.
reconstruction error=0.7208009362220764, variation=2.980232238769531e-07.
reconstruction error=0.7208007574081421, variation=1.7881393432617188e-07.
reconstruction error=0.7208005785942078, variation=1.7881393432617188e-07.
reconstruction error=0.7208003401756287, variation=2.384185791015625e-07.
reconstruction error=0.7208001613616943, variation=1.7881393432617188e-07.
reconstruction error=0.72079998254776, variation=1.7881393432617188e-07.
reconstruction error=0.7207998037338257, variation=1.7881393432617188e-07.
reconstruction error=0.7207996845245361, variation=1.1920928955078125e-07.
reconstruction error=0.7207995057106018, variation=1.7881393432617188e-07.
reconstruction error=0.7207993865013123, variation=1.1920928955078125e-07.
reconstruction error=0.7207992076873779, variation=1.7881393432617188e-07.
reconstruction error=0.7207990288734436, variation=1.7881393432617188e-07.
reconstruction error=0.720798909664154, variation=1.1920928955078125e-07.
reconstruction error=0.7207988500595093, variation=5.960464477539063e-08.
reconstruction error=0.7207987308502197, variation=1.1920928955078125e-07.
reconstruction error=0.7207986116409302, variation=1.1920928955078125e-07.
reconstruction error=0.7207984924316406, variation=1.1920928955078125e-07.
reconstruction error=0.7207984328269958, variation=5.960464477539063e-08.
reconstruction error=0.7207983136177063, variation=1.1920928955078125e-07.
reconstruction error=0.7207981944084167, variation=1.1920928955078125e-07.
reconstruction error=0.720798134803772, variation=5.960464477539063e-08.
reconstruction error=0.7207980751991272, variation=5.960464477539063e-08.
reconstruction error=0.7207979559898376, variation=1.1920928955078125e-07.
reconstruction error=0.7207978963851929, variation=5.960464477539063e-08.
reconstruction error=0.7207978963851929, variation=0.0.
converged in 144 iterations.
Mode A shape: (579, 3)
Mode B shape: (5, 3)
Mode C shape: (3, 3)
Rank A: 3
Rank B: 3
Rank C: 3
Corrcoef A:
 [[ 1.         -0.15515206 -0.02237843]
 [-0.15515206  1.          0.27849014]
 [-0.02237843  0.27849014  1.        ]]
Corrcoef B:
 [[ 1.         -0.15171797  0.09300134]
 [-0.15171797  1.         -0.15544876]
 [ 0.09300134 -0.15544876  1.        ]]
Corrcoef C:
 [[ 1.         -0.95836522  0.49442482]
 [-0.95836522  1.         -0.72204133]
 [ 0.49442482 -0.72204133  1.        ]]
Mode C rank-1 vector values:
 [[0.7413153  0.41862255 0.39879617]
 [0.1774483  0.7133976  0.13733484]
 [0.49918738 0.46672788 0.7542782 ]]

====== w2v_seed=0, w2v_dim=5, cpd_rs=1 ======
kv_jpop.vectors.shape: (579, 5)
stacked X shape: (579, 5, 3)
reconstruction error=0.7873014807701111, variation=0.012822210788726807.
reconstruction error=0.7739747166633606, variation=0.013326764106750488.
reconstruction error=0.7623987197875977, variation=0.01157599687576294.
reconstruction error=0.7542356848716736, variation=0.008163034915924072.
reconstruction error=0.7486257553100586, variation=0.00560992956161499.
reconstruction error=0.7443550229072571, variation=0.004270732402801514.
reconstruction error=0.7407357692718506, variation=0.003619253635406494.
reconstruction error=0.7375034689903259, variation=0.003232300281524658.
reconstruction error=0.7346176505088806, variation=0.0028858184814453125.
reconstruction error=0.732120156288147, variation=0.0024974942207336426.
reconstruction error=0.7300454378128052, variation=0.002074718475341797.
reconstruction error=0.7283833622932434, variation=0.0016620755195617676.
reconstruction error=0.7270832657814026, variation=0.0013000965118408203.
reconstruction error=0.7260748147964478, variation=0.001008450984954834.
reconstruction error=0.7252873778343201, variation=0.0007874369621276855.
reconstruction error=0.7246608138084412, variation=0.0006265640258789062.
reconstruction error=0.724149227142334, variation=0.0005115866661071777.
reconstruction error=0.7237193584442139, variation=0.0004298686981201172.
reconstruction error=0.7233483791351318, variation=0.00037097930908203125.
reconstruction error=0.7230210900306702, variation=0.0003272891044616699.
reconstruction error=0.7227274775505066, variation=0.0002936124801635742.
reconstruction error=0.7224610447883606, variation=0.0002664327621459961.
reconstruction error=0.7222175598144531, variation=0.0002434849739074707.
reconstruction error=0.7219944000244141, variation=0.0002231597900390625.
reconstruction error=0.7217897176742554, variation=0.0002046823501586914.
reconstruction error=0.721602201461792, variation=0.0001875162124633789.
reconstruction error=0.7214308977127075, variation=0.00017130374908447266.
reconstruction error=0.7212750315666199, variation=0.00015586614608764648.
reconstruction error=0.7211340069770813, variation=0.00014102458953857422.
reconstruction error=0.7210069298744202, variation=0.0001270771026611328.
reconstruction error=0.7208933234214783, variation=0.00011360645294189453.
reconstruction error=0.7207922339439392, variation=0.0001010894775390625.
reconstruction error=0.7207030653953552, variation=8.916854858398438e-05.
reconstruction error=0.7206249833106995, variation=7.808208465576172e-05.
reconstruction error=0.7205572724342346, variation=6.771087646484375e-05.
reconstruction error=0.7204990386962891, variation=5.823373794555664e-05.
reconstruction error=0.7204495668411255, variation=4.947185516357422e-05.
reconstruction error=0.7204080820083618, variation=4.1484832763671875e-05.
reconstruction error=0.7203739285469055, variation=3.415346145629883e-05.
reconstruction error=0.7203462719917297, variation=2.765655517578125e-05.
reconstruction error=0.7203245759010315, variation=2.1696090698242188e-05.
reconstruction error=0.7203080654144287, variation=1.6510486602783203e-05.
reconstruction error=0.7202962040901184, variation=1.1861324310302734e-05.
reconstruction error=0.7202884554862976, variation=7.748603820800781e-06.
reconstruction error=0.7202842831611633, variation=4.172325134277344e-06.
reconstruction error=0.7202832698822021, variation=1.0132789611816406e-06.
reconstruction error=0.7202850580215454, variation=-1.7881393432617188e-06.
converged in 48 iterations.
Mode A shape: (579, 3)
Mode B shape: (5, 3)
Mode C shape: (3, 3)
Rank A: 3
Rank B: 3
Rank C: 3
Corrcoef A:
 [[ 1.         -0.10001576  0.05298846]
 [-0.10001576  1.         -0.23335455]
 [ 0.05298846 -0.23335455  1.        ]]
Corrcoef B:
 [[ 1.         -0.19301981 -0.06393634]
 [-0.19301981  1.          0.12301479]
 [-0.06393634  0.12301479  1.        ]]
Corrcoef C:
 [[ 1.         -0.96302421  0.50649153]
 [-0.96302421  1.         -0.72006521]
 [ 0.50649153 -0.72006521  1.        ]]
Mode C rank-1 vector values:
 [[0.73947406 0.42155826 0.39432478]
 [0.18678918 0.7129003  0.11868221]
 [0.50540036 0.47200936 0.7596022 ]]

====== w2v_seed=0, w2v_dim=5, cpd_rs=2 ======
kv_jpop.vectors.shape: (579, 5)
stacked X shape: (579, 5, 3)
reconstruction error=0.7750279903411865, variation=0.007918894290924072.
reconstruction error=0.7726383209228516, variation=0.002389669418334961.
reconstruction error=0.7716478705406189, variation=0.000990450382232666.
reconstruction error=0.770796537399292, variation=0.0008513331413269043.
reconstruction error=0.7697276473045349, variation=0.00106889009475708.
reconstruction error=0.7683603763580322, variation=0.0013672709465026855.
reconstruction error=0.7666323184967041, variation=0.001728057861328125.
reconstruction error=0.7643696069717407, variation=0.002262711524963379.
reconstruction error=0.7612407803535461, variation=0.00312882661819458.
reconstruction error=0.7568389773368835, variation=0.004401803016662598.
reconstruction error=0.7509762644767761, variation=0.005862712860107422.
reconstruction error=0.7440932989120483, variation=0.006882965564727783.
reconstruction error=0.7372887134552002, variation=0.0068045854568481445.
reconstruction error=0.7316542863845825, variation=0.005634427070617676.
reconstruction error=0.7276170253753662, variation=0.004037261009216309.
reconstruction error=0.724982500076294, variation=0.0026345252990722656.
reconstruction error=0.7233393788337708, variation=0.0016431212425231934.
reconstruction error=0.7223256230354309, variation=0.0010137557983398438.
reconstruction error=0.7216944694519043, variation=0.0006311535835266113.
reconstruction error=0.7212948799133301, variation=0.00039958953857421875.
reconstruction error=0.7210373282432556, variation=0.0002575516700744629.
reconstruction error=0.7208690643310547, variation=0.00016826391220092773.
reconstruction error=0.7207581996917725, variation=0.00011086463928222656.
reconstruction error=0.7206850647926331, variation=7.31348991394043e-05.
reconstruction error=0.7206372022628784, variation=4.786252975463867e-05.
reconstruction error=0.7206066846847534, variation=3.0517578125e-05.
reconstruction error=0.7205881476402283, variation=1.8537044525146484e-05.
reconstruction error=0.7205779552459717, variation=1.0192394256591797e-05.
reconstruction error=0.7205734252929688, variation=4.5299530029296875e-06.
reconstruction error=0.7205730676651001, variation=3.5762786865234375e-07.
reconstruction error=0.7205755710601807, variation=-2.5033950805664062e-06.
converged in 32 iterations.
Mode A shape: (579, 3)
Mode B shape: (5, 3)
Mode C shape: (3, 3)
Rank A: 3
Rank B: 3
Rank C: 3
Corrcoef A:
 [[ 1.         -0.13938336  0.03156151]
 [-0.13938336  1.         -0.2665223 ]
 [ 0.03156151 -0.2665223   1.        ]]
Corrcoef B:
 [[ 1.         -0.16541505 -0.08503388]
 [-0.16541505  1.          0.14527315]
 [-0.08503388  0.14527315  1.        ]]
Corrcoef C:
 [[ 1.         -0.95981821  0.49751348]
 [-0.95981821  1.         -0.72095026]
 [ 0.49751348 -0.72095026  1.        ]]
Mode C rank-1 vector values:
 [[0.7407955  0.41963705 0.39673218]
 [0.18056017 0.71327287 0.13080864]
 [0.5011533  0.4685028  0.7561427 ]]

Process finished with exit code 0

'''