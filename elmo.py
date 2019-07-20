from elmoformanylangs import Embedder
import pickle
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import nltk

nltk.download("stopwords")
import string
from nltk.corpus import stopwords
import numpy as np
import pickle
import re


def load_data():
    with open("data.pkl", "rb") as f:
        corpus = pickle.load(f)
    return corpus


# **********************************PRE-PROCESSING AND TRANSLITERATE*****************


def remove_links():
    corpus_clean = []
    for l, i in enumerate(corpus):
        flag = [0, 0]
        for j, k in enumerate(i[:-1]):
            if k[0] == "twitter":
                try:
                    temp = k[0] + i[j + 1][0] + i[j + 2][0][:3]
                    if temp == "twitter.com":
                        try:
                            if i[j - 2][0][-3:] == "pic" and len(i[j - 2][0]) != 3:
                                flag = [2, j]
                            elif i[j - 2][0] == "pic":
                                flag = [1, j]
                        except:
                            pass
                except:
                    pass
        if flag[0] == 0:
            corpus_clean.append(i)
        elif flag[0] == 1:
            corpus_clean.append(i[: flag[1] - 2] + i[flag[1] + 3 :])
        else:
            new = (i[flag[1] - 2][0], "Hi")
            corpus_clean.append(i[: flag[1] - 2] + [new] + i[flag[1] + 3 :])
    corpus_clean1 = []
    for d, i in enumerate(corpus_clean):
        flag = [0, 0]
        for l, j in enumerate(i[:-1]):
            if "twitter.com" in j[0]:
                flag = [1, l]
        if flag[0] == 0:
            corpus_clean1.append(i)
        else:
            new = i[flag[1]][0].split("pic")
            if new[0] == "":
                corpus_clean1.append(i[: flag[1]] + i[flag[1] + 1 :])
            else:
                new = (new[0], i[flag[1]][1])
                corpus_clean1.append(i[: flag[1]] + [new] + i[flag[1] + 1 :])
    corpus_clean2 = []
    for i in corpus_clean1:
        flag = [0, 0]
        for k, j in enumerate(i[:-1]):
            if "pic" in j[0] and j[1] == "Hi":
                temp = j[0].split("pic")
                if temp[0] != "" and temp[-1] == "":
                    flag = [1, k]
                    break
        if flag[0] == 0:
            corpus_clean2.append(i)
        else:
            temp = i[flag[1]][0].split("pic")
            corpus_clean2.append(i[: flag[1]] + [(temp[0], "Hi")] + i[flag[1] + 1 :])
    corpus_clean3 = []
    for i in corpus_clean2:
        corpus_clean3.append(i[:-1])

    return corpus_clean3


def remove_empty():
    corpus1 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[0] == "":
                pass
            else:
                temp.append(j)
        corpus1.append(temp)
    return corpus1


def remove_punc_lower():
    corpus2 = []
    x = lambda e: (e[0].lower(), e[1])
    punc = list(string.punctuation)
    punc.remove("#")
    punc.remove("@")
    corpus1 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1] == "Ot" and j[0][0] == "#":
                q = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", j[0][1:])
                for i in q.split(" "):
                    temp.append((i, "En"))
            elif j[1] != "Ot":
                temp.append(j)
        corpus1.append(temp)
    for i in corpus1:
        temp = []
        for j in i:
            temp.append(x(j))
        corpus2.append(temp)
    return corpus2


def trans_literate():
    with open("hindi_dict.txt") as f:
        hindi_trans = f.read()
    hindi_trans = hindi_trans.split("\n")
    hindi_trans = [i for i in hindi_trans if i != ""]
    hindi_trans = [[i for i in j.split(" ") if i != ""] for j in hindi_trans]
    for i in hindi_trans:
        if len(i) != 2:  # unusable words
            hindi_trans.remove(i)
    hindi_trans_dict = {}
    for i in hindi_trans:
        hindi_trans_dict[i[0]] = i[1]
    corpus_trans = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1] == "Hi" and j[0] in hindi_trans_dict:
                temp.append((hindi_trans_dict[j[0]], "Hi"))
            else:
                temp.append((j[0], "En"))
        corpus_trans.append(temp)
    return corpus_trans


def remove_stopwords():
    #!wget 'https://raw.githubusercontent.com/Alir3z4/stop-words/master/hindi.txt'
    with open("hindi.txt") as f:
        stopwords_hi = f.read()
    stop_words_hi = set(stopwords_hi.split("\n"))
    stop_words_en = set(stopwords.words("english"))
    corpus2 = []
    for i in corpus_new:
        temp = []
        for j in i:
            if j[1] == "Hi" and j[0] in stop_words_hi:
                print(j)
                pass
            elif j[1] == "En" and j[0] in stop_words_en:
                print(j)
                pass
            else:
                temp.append(j)
        corpus2.append(temp)
    return corpus2


def sanity_check():
    for i in corpus_new:
        if len(i) == 0:
            print("CHECK LEN")


def ge():
    embeddings = []
    for i, j in enumerate(corpus_new):
        print(
            "===========================================",
            i,
            "===========================================",
        )
        flag = j[0][1]
        temp = []
        sents = []
        for k, l in enumerate(j):
            if l[1] != flag:
                print(sents)
                if flag == "Hi":
                    temp.append(emb_hi.sents2elmo([sents], output_layer=-1))
                else:
                    temp.append(emb_en.sents2elmo([sents], output_layer=-1))
                flag = l[1]
                sents = [l[0]]
            else:
                sents.append(l[0])
            if len(j) == k + 1:
                print(sents)
                if flag == "Hi":
                    temp.append(emb_hi.sents2elmo([sents], output_layer=-1))
                else:
                    temp.append(emb_en.sents2elmo([sents], output_layer=-1))
            temps = []
        for i in temp:
            if len(i[0]) != 1:
                for j in i[0]:
                    temps.append(j)
            else:
                temps.append(i[0][0])
        embeddings.append(temps)
    return embeddings


def intitialize_embeddings():
    emb_en = Embedder("144")
    emb_hi = Embedder("155")
    return emb_hi, emb_en


corpus = load_data()
corpus_new = remove_links()
corpus_new = remove_empty()
corpus_new = remove_punc_lower()
corpus_new = remove_empty()
corpus_new = trans_literate()
corpus_new = remove_empty()
corpus_new = remove_stopwords()
corpus_new = remove_empty()
sanity_check()
print("1 done")
print(1)
emb_hi, emb_en = intitialize_embeddings()
embeddings = ge()
print("2 done")

embeddings = np.array(embeddings)
print(embeddings)
np.save("elmo_final.npy", embeddings)
