from sklearn.metrics import silhouette_score
from collections import Counter

def silhouette(X, labels):
    return silhouette_score(X, labels)


def cluster_distribution(labels):
    return Counter(labels)
