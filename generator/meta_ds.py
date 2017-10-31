import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
pd.set_option('expand_frame_repr', False)

n = 1000
m = 10

data_ = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
         [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

data = pd.DataFrame(data_)

# add empty column for the labels
data['label'] = np.nan

print(data)

print('Calculate Distance Matrix...')
dist = np.triu(
    distance.cdist(data[data[:].columns.difference(['label'])], data[data[:].columns.difference(['label'])],
                   'euclidean'))
print(dist)
# calculate Minimum Spanning Tree
print('Calculate MST...')
mst = minimum_spanning_tree(dist, overwrite=False).toarray()

# get row and column indices of non-zero values in mst
# mst_edges has this form: [[0, 0], [1, 3], [1, 4], [3, 4]]
print('Get row and column indices of non-zero values in MST...')
# noinspection PyTypeChecker
mst_edges = (np.argwhere(mst != 0)).tolist()
print(mst)
print(mst_edges)
print(len(mst_edges[:]))
