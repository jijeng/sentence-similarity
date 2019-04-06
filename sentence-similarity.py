import pandas as pd
import numpy as np
import scipy
import math
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


## data preprocessing
def load_sts_dataset(filename):
    sent_pairs = []
    # tensorflow file mapunication
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[5], ts[6], float(ts[4])))

    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])


def download_and_load_sts_data():
    base_path = "./"

    sts_dev = load_sts_dataset(os.path.join(os.path.dirname(base_path), "stsbenchmark", "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(os.path.dirname(base_path), "stsbenchmark", "sts-test.csv"))

    return sts_dev, sts_test


sts_dev, sts_test = download_and_load_sts_data()


def download_sick1(f):
    lines = []
    file = open(f, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        line = line.split('\t')
        if len(line) != 5:
            continue
        tmp = line[4].replace('\n', '')
        tmp2 = line[:4]
        tmp2.append(tmp)
        lines.append(tmp2)

    lines = lines[1:]
    df = pd.DataFrame(lines, columns=['idx', 'sent_1', 'sent_2', 'sim', 'label'])
    df['sim'] = pd.to_numeric(df['sim'])
    return df


sick_train = download_sick1("./data/SICK_train.txt")
sick_dev = download_sick1("./data/SICK_trial.txt")
sick_test = download_sick1("./data/SICK_test_annotated.txt")

sick_all = sick_train.append(sick_test).append(sick_dev)

import nltk

# nltk.download('stopwords') # if you don't have , uncommment it
STOP = set(nltk.corpus.stopwords.words("english"))


class Sentence:

    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]


# pre-trained model
import gensim

from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec

# PATH_TO_WORD2VEC = os.path.expanduser("~/nlp/GoogleNews-vectors-negative300.bin")
# PATH_TO_GLOVE = os.path.expanduser("~/nlp/glove.840B.300d.txt")

PATH_TO_WORD2VEC = os.path.expanduser("~/sentence-similarity/model/GoogleNews-vectors-negative300.bin")
PATH_TO_GLOVE = os.path.expanduser("~/sentence-similarity/mdoel/glove.840B.300d.txt")


# 这个load的过程还是相当的慢的
word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)

tmp_file = "/tmp/glove.840B.300d.w2v.txt"

# glove2word2vec(PATH_TO_GLOVE, tmp_file) # uncomment it if first run

glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)


import csv

PATH_TO_FREQUENCIES_FILE = os.path.expanduser("~/sentence-similarity/data/frequencies.tsv")
PATH_TO_DOC_FREQUENCIES_FILE = os.path.expanduser("~/sentence-similarity/data/doc_frequencies.tsv")


def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies


frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE)
doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
doc_frequencies["NUM_DOCS"] = 1288431

# comparision metrics
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


def run_avg_benchmark(sentences1, sentences2, model=None, use_stoplist=False, doc_freqs=None):
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]

    sims = []
    for (sent1, sent2) in zip(sentences1, sentences2):

        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)

        weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs1] if doc_freqs else None
        weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs2] if doc_freqs else None

        embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

    return sims


# Smooth Inverse Frequency implementation
from sklearn.decomposition import TruncatedSVD


def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def run_sif_benchmark(sentences1, sentences2, model, freqs={}, use_stoplist=False, a=0.001):
    total_freq = sum(freqs.values())

    embeddings = []

    # SIF requires us to first collect all sentence embeddings and then perform
    # common component analysis.
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        weights1 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens1]
        weights2 = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens2]

        # import ipdb
        # ipdb.set_trace()

        embedding1 = np.average([model[token] for token in tokens1], axis=0, weights=weights1)
        embedding2 = np.average([model[token] for token in tokens2], axis=0, weights=weights2)

        embeddings.append(embedding1)
        embeddings.append(embedding2)

    embeddings = remove_first_principal_component(np.array(embeddings))
    sims = [cosine_similarity(embeddings[idx * 2].reshape(1, -1),
                              embeddings[idx * 2 + 1].reshape(1, -1))[0][0]
            for idx in range(int(len(embeddings) / 2))]

    return sims


import functools as ft

benchmarks = [
    ("AVG-GLOVE", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False)),
    ("AVG-GLOVE-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True)),
    ("AVG-GLOVE-TFIDF", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False, doc_freqs=doc_frequencies)),
    ("AVG-GLOVE-TFIDF-STOP", ft.partial(run_avg_benchmark, model=glove, use_stoplist=True, doc_freqs=doc_frequencies)),
    ("SIF-W2V", ft.partial(run_sif_benchmark, freqs=frequencies, model=word2vec, use_stoplist=False)),
    ("SIF-GLOVE", ft.partial(run_sif_benchmark, freqs=frequencies, model=glove, use_stoplist=False)),

]

pearson_results, spearman_results = {}, {}


# nltk.download('punkt') if first run

def run_experiment(df, benchmarks):
    sentences1 = [Sentence(s) for s in df['sent_1']]
    sentences2 = [Sentence(s) for s in df['sent_2']]

    pearson_cors, spearman_cors = [], []
    for label, method in benchmarks:
        sims = method(sentences1, sentences2)
        pearson_correlation = scipy.stats.pearsonr(sims, df['sim'])[0]
        print(label, pearson_correlation)
        pearson_cors.append(pearson_correlation)
        # 两个相关系数
        spearman_correlation = scipy.stats.spearmanr(sims, df['sim'])[0]
        spearman_cors.append(spearman_correlation)

    return pearson_cors, spearman_cors


pearson_results["SICK-DEV"], spearman_results["SICK-DEV"] = run_experiment(sick_dev, benchmarks)
pearson_results["SICK-TEST"], spearman_results["SICK-TEST"] = run_experiment(sick_test, benchmarks)

pearson_results["STS-DEV"], spearman_results["STS-DEV"] = run_experiment(sts_dev, benchmarks)
pearson_results["STS-TEST"], spearman_results["STS-TEST"] = run_experiment(sts_test, benchmarks)



pearson_results_df = pd.DataFrame(pearson_results)
pearson_results_df = pearson_results_df.transpose()

pearson_results_df = pearson_results_df.rename(columns={i: b[0] for i, b in enumerate(benchmarks)})

spearman_results_df = pd.DataFrame(spearman_results)
spearman_results_df = spearman_results_df.transpose()
spearman_results_df = spearman_results_df.rename(columns={i: b[0] for i, b in enumerate(benchmarks)})

#
pearson_results_df.to_csv('pearson_results.csv')
spearman_results_df.to_csv('spearman_results.csv')
print('success! ')

