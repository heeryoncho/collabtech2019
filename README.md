# Discovering Latent Country Words: A Step towards Cross-cultural Emotional Communication
Source code and data for CollabTech 2019 paper [Discovering Latent Country Words: A Step towards Cross-cultural Emotional Communication](https://drive.google.com/file/d/1NKaS4dx2iaCU4e1XrOMmrQkgYHKe4WXk/view) by Heeryon Cho and Toru Ishida.
This code utlizes word embedding and partially fixed CP (CANDECOMP/PARAFAC) decomposition to find latent country words from 10 years worth (2008--2017) of J-pop and K-pop yearly top-100 ranking lyrics.

![](https://github.com/heeryoncho/collabtech2019/blob/master/fig/tensor.png)

![](https://github.com/heeryoncho/collabtech2019/blob/master/fig/table7.png)
![](https://github.com/heeryoncho/collabtech2019/blob/master/fig/table8.png)
![](https://github.com/heeryoncho/collabtech2019/blob/master/fig/table9.png)

## Requirements
The code runs with:
* Ubuntu 16.04
* Python 3.5.2

## Libraries
Various Python libraries are required including:
* tensorly 0.4.2
* gensim 3.5.0

## Citation
If you find this useful, please cite our work as follows:
```
@proceedings{ChoIshida_CollabTech2019,
  author    = {Heeryon Cho and Toru Ishida},
  title     = {Discovering Latent Country Words: A Step towards Cross-cultural Emotional Communication},
  booktitle = {Proceedings of CollabTech 2019},
  volume    = {XXXX},
  series    = {LNCS},
  publisher = {Springer},
  year      = {2019},
}
```
