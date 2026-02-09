from collections import Counter
from itertools import combinations
import math

def extract_cooccurrences(tokenized_docs, min_freq=3):
    cooc = Counter()
    for doc in tokenized_docs:
        tokens = set(doc)
        for pair in combinations(tokens, 2):
            cooc[pair] += 1
    return {pair: freq for pair, freq in cooc.items() if freq >= min_freq}


def compute_pmi(tokenized_docs, min_freq=3):
    total_docs = len(tokenized_docs)
    word_freq = Counter()
    pair_freq = Counter()

    for doc in tokenized_docs:
        tokens = set(doc)
        for w in tokens:
            word_freq[w] += 1
        for pair in combinations(tokens, 2):
            pair_freq[pair] += 1

    pmi_scores = {}
    for (w1, w2), freq in pair_freq.items():
        if freq >= min_freq:
            p_w1 = word_freq[w1] / total_docs
            p_w2 = word_freq[w2] / total_docs
            p_w1_w2 = freq / total_docs
            pmi = math.log2(p_w1_w2 / (p_w1 * p_w2))
            pmi_scores[(w1, w2)] = pmi

    return pmi_scores
