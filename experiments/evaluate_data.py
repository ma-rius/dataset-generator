"""
This script can be used to calculate the complexity of an existing data set
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance

np.set_printoptions(threshold=np.inf, precision=10, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)

path = '../assets/complexity_0.2/data_1.csv'
m_subs_i = 0
m_subs_j = 15

data_ = pd.read_csv(filepath_or_buffer=path, usecols=[x for x in range(m_subs_i, m_subs_j, 1)])

print(data_)
print()

# build distance matrix
dist = np.triu(
    distance.cdist(data_[data_[:].columns.difference(['label'])], data_[data_[:].columns.difference(['label'])],
                   'euclidean'))

# calculate Minimum Spanning Tree
mst = minimum_spanning_tree(dist, overwrite=False).toarray()

# get row and column indices of non-zero values in mst
# mst_edges has this form: [[0, 0], [1, 3], [1, 4], [3, 4]]
# noinspection PyTypeChecker
mst_edges = (np.argwhere(mst != 0)).tolist()
# print(mst)
# print(mst_edges)

individual = pd.read_csv(filepath_or_buffer=path, usecols=['label'])

print(np.array(individual['label'].as_matrix()))
print()


# calculate complexity measure
def complexity(individual, mst_edges, n):
    # 1. Store the nodes of the spanning tree with different class.
    nodes = [-1] * n
    for edge in mst_edges:
        if individual[edge[0]] != individual[edge[1]]:
            nodes[edge[0]] = 0
            nodes[edge[1]] = 0

    # 2. Compute the number of nodes of the spanning tree with different class.
    different = 0
    for i in range(n):
        if nodes[i] == 0:
            different += 1

    complexity_ = different / n
    return complexity_


print(complexity(individual=np.array(individual['label'].as_matrix()), mst_edges=mst_edges, n=len(data_)))
