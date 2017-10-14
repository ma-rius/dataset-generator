import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.spatial.distance as dist


# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)

P = np.random.randint(10, size=(10, 2))
print(P)
print()
d = np.triu(cdist(P, P))
print(d)
print()
print(minimum_spanning_tree(d).toarray())
# a = np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
#
# print(a)

for p in range(10):
    row = []
    for q in range(0, p + 1, 1):
        row.append(0)
    for q in range(p + 1, 10, 1):
        u, v = P[0], P[1]
        d = dist.euclidean(u, v)
        # d = np.linalg.norm(np.array(u) - np.array(v))  # about twice as fast as dist.euclidean
        row.append(d)
    print(row)