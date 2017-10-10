import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

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