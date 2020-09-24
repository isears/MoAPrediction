from sklearn.cluster import KMeans
from LocalUtil import *


X, Y = load_cache('./data/cache/X.npy', './data/cache/Y.npy')

all_scores = list()

for idx in range(2, 10000):
    kmeans = KMeans(n_clusters=idx).fit(X)
    this_score = kmeans.score(X)
    all_scores.append(this_score)
    print(f'Score for {idx} clusters: {this_score}')


print(all_scores)