from flair.embeddings import BertEmbeddings, WordEmbeddings
from flair.data import Sentence
import pickle
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from elmoformanylangs import Embedder
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
import string

# ---------------------------------------PREPROCESSING--------------------------------------#


def load_data():
    with open("../data/data.pkl", "rb") as f:
        corpus = pickle.load(f)
    return corpus


def indic_trans():
    corpus_new = []
    scheme_map = SchemeMap(SCHEMES[sanscript.ITRANS], SCHEMES[sanscript.DEVANAGARI])
    for i in corpus:
        temp = []
        for j in i:
            if type(j) != int:
                if j[1] == "Hi":
                    temp.append((transliterate(j[0], scheme_map=scheme_map), j[1]))
                else:
                    temp.append((j[0], j[1]))
        corpus_new.append(temp)
    return corpus_new


def remove_stopwords():
    with open("hindi.txt") as f:
        stopwords_hi = f.read()
    stop_words_hi = set(stopwords_hi.split("\n"))
    stop_words = set.union(set(stopwords.words("english")), stop_words_hi)
    for i in range(len(corpus_trans)):
        for j, k in enumerate(corpus_trans[i]):
            if k[0] in stop_words:
                a = corpus_trans[i].pop(j)
    return corpus_trans


def remove_punctuation():
    punc = list(string.punctuation)
    punc.remove("#")
    punc.remove("@")
    indices = []
    for i, j in enumerate(corpus_trans):
        index = []
        for k, l in enumerate(j):
            if l[1] == "Ot":
                if l[0][0] in punc:
                    index.append(k)
                    # corpus_trans[i].pop(k)
        indices.append(index)
    for i, j in enumerate(indices):
        for index in sorted(j, reverse=True):
            assert corpus_trans[i][index][0][0] in punc
            del corpus_trans[i][index]
    return corpus_trans


def convert_tags_hashtags():
    for i, j in enumerate(corpus_trans):
        for k, l in enumerate(j):
            if l[1] == "Ot":
                if l[0][0] in "#@":
                    corpus_trans[i][k] = (
                        corpus_trans[i][k][0][1:],
                        corpus_trans[i][k][1],
                    )
    return corpus_trans


# -------------------------------------------EMBEDDINGS----------------------------------------#

# ---------------------------FASTTEXT-------------------------------#


def initialize_embeddings_fasttext():
    embedding_fasttext_hi = WordEmbeddings("hi")
    embedding_fasttext_en = WordEmbeddings("en")
    return embedding_fasttext_hi, embedding_fasttext_en


def get_embeddings_fasttext():
    embeddings_fasttext_temp = []
    for i in corpus_trans:
        temp = []
        for j in i:
            try:
                if j[1] != "En":
                    temp.append(embedding_fasttext_hi.embed(Sentence(j[0])))
                else:
                    temp.append(embedding_fasttext_en.embed(Sentence(j[0])))
            except:
                print(j)
        embeddings_fasttext_temp.append(temp)

    embeddings_fasttext = []
    for i in embeddings_fasttext_temp:
        temp = []
        for j in i:
            for k in j:
                a = [np.array(token.embedding) for token in k]
            temp.append(a[0].tolist())
        embeddings_fasttext.append(temp)
    return embeddings_fasttext


# ---------------------------ELMoForManyLangs-----------------------------#


def intitialize_embeddings_elmo():
    emb_en = Embedder("144")
    emb_hi = Embedder("155")
    return emb_hi, emb_en


def generate_embeddings():
    embeddings = []
    for i, j in enumerate(corpus_trans[:5]):
        temp = []
        for k, l in enumerate(j):
            if l[1] == "Hi":
                temp.append(emb_hi.sents2elmo(l[0], output_layer=-1))
            else:
                temp.append(emb_en.sents2elmo(l[0], output_layer=-1))
        embeddings.append(temp)
    return embeddings


# ---------------------------------BERT-----------------------------------#


def initialize_embeddings_bert():
    embedding_bert = BertEmbeddings("bert-base-multilingual-cased")
    return embedding_bert


def get_embeddings_bert():
    embeddings = []
    for t, i in enumerate(corpus_trans):
        sentence = Sentence((" ").join([j[0] for j in corpus_trans[0]]))
        embeddings.append(embedding_bert.embed(sentence))
    embeddings_bert = []
    for t, i in enumerate(embeddings):
        temp = []
        for j in i:
            for k in j:
                a = [np.array(token.embedding) for token in k]
            temp.append(a[0].tolist())
        embeddings_bert.append(temp)
    return embeddings_bert


# ---------------------------------------------ANALYZE-----------------------------------------#


def check_others():
    other = {}
    for i in corpus_trans:
        for j in i:
            if type(j) != int:
                if j[1] == "Ot":
                    if j[0] in other:
                        other[j[0]] += 1
                    else:
                        other[j[0]] = 1
    print(sorted(other.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))


def analyze_embeddings(emb):
    """in format hindi, english, others, total"""
    dic = {"Hi": 0, "En": 1, "Ot": 2}
    count = [0, 0, 0, 0]
    count_zero = [0, 0, 0, 0]
    for i, j in zip(emb, corpus_trans):
        for k, l in zip(i, j):
            count[dic[l[1]]] += 1
            if sum(k) == 0:
                count_zero[dic[l[1]]] += 1
    count[-1] = sum(count)
    count_zero[-1] - sum(count_zero)
    print("hi, en, ot, total")
    print("count: ", count)
    print("zero count: ", count_zero)


# ----------------------------------------------MAIN-------------------------------------------#
corpus = load_data()
print("Corpus Loaded")
corpus_trans = indic_trans()
corpus_trans = remove_stopwords()
corpus_trans = remove_punctuation()
corpus_trans = convert_tags_hashtags()
print("Corpus Processed")


print("Generating fasttext embeddings")
# FASTTEXT
embedding_fasttext_hi, embedding_fasttext_en = initialize_embeddings_fasttext()
embeddings_fasttext = get_embeddings_fasttext()
np.save("fasttext_multi.npy", np.array(embeddings_fasttext))
print("Fasttext embeddings generated")

print("Generating ELMO embeddings")
# ELMO
emb_hi, emb_en = intitialize_embeddings_elmo()
embeddings_elmo = generate_embeddings()
np.save("elmo_multi.npy", np.array(embeddings_elmo))
print("ELMO embeddings generated")

print("Generating BERT embeddings")
# BERT
embedding_bert = initialize_embeddings_bert()
emb = get_embeddings_bert()
np.save("bert_multi.npy", np.array(emb))
print("BERT embeddings generated")
