import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph._min_spanning_tree import minimum_spanning_tree
from scipy.spatial import distance

m = 200
n = 10000

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)

# d = {'one': pd.Series([1., 2., 3.]),
#      'two': pd.Series([1., 2., 3., 4.])}
# df = pd.DataFrame(d)
# print(df)
# print()

# def func():
#     dfs = [pd.DataFrame(np.random.randn(size_per, N))
#            for _ in range(N)]
#     return pd.concat(dfs, ignore_index=True)


# s = pd.DataFrame([[1, 2, 3], [2, 4, 5]])

# print(s.info())

# noinspection PyTypeChecker
# dist = distance.cdist(df, df, 'euclidean')
# print(dist)

start = time.time()
mean = np.random.randint(100, size=m)
A = np.random.rand(m, m)
cov = np.dot(A, A.transpose())

s = pd.DataFrame(np.random.multivariate_normal(mean, cov, n))
end = time.time()
print('------------------')
print('Time of Data Set calculation: ', (end-start), 'seconds')
print('------------------')
print('INFO DATASET')
print(s.info())
print('------------------')


start = time.time()
# noinspection PyTypeChecker
dist = distance.cdist(s, s, 'euclidean')
dist = pd.DataFrame(dist)
end = time.time()
print('------------------')
print('Time of Distance Matrix calculation: ', (end-start), 'seconds')
print('------------------')
print('INFO DISTANCE MATRIX')
print(dist.info())
print('------------------')

start = time.time()
# noinspection PyTypeChecker
mst = minimum_spanning_tree(dist, overwrite=True).toarray()
end = time.time()

print('------------------')
print('Time of MST calculation: ', (end-start), 'seconds')

