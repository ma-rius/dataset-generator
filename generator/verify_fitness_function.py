"""
This script was used to verify the fitness function (i.e. the complexity of a given dataset).
It generates a random data set and calculates the complexity.
This dataset could then be used to check if the complexity was the same as the one calculated by
the "Data Complexity Library in C++" (Albert Orriols-Puig, Núria Macià, and Tin Kam Ho).
"""
import csv
import random
from random import randint
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
from joblib import Parallel, delayed
from progressbar import ProgressBar
from scipy.sparse.csgraph import minimum_spanning_tree

# -------- Dataset Parameters --------
m = 5  # number of features
n = 10  # number of instances
b = 1 / n  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].

# -------- End Dataset Parameters --------


# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)

# -------- create and store dataset without labels --------
distribution_dict = {}
instances = {}

print('initialize distribution')
progress_bar_distribution = ProgressBar()
for j in progress_bar_distribution(range(0, m, 1)):
    # set distribution for feature j
    mu = random.randint(0, 100)
    sigma = random.randint(0, 100)
    # print('Mu: %d, Sigma: %s' % (mu, sigma))
    distribution_dict.update({j: [mu, sigma]})

print('randomly pick values from distributions')
progress_bar_random = ProgressBar()
for i in progress_bar_random(range(0, n, 1)):
    instance = []
    for j in range(0, m, 1):
        instance.append(np.random.normal(distribution_dict[j][0], distribution_dict[j][1], 1)[0])
    instances.update({i: instance})

# store data in csv file
with open('../assets/verify_data.csv', 'w') as f:
    print('write instances to file')
    progress_bar_file = ProgressBar()
    wtr = csv.writer(f, delimiter=',')
    # create top line in csv
    top_line = []
    for title in range(m):
        top_line.append(title)
    top_line.append('label')
    wtr.writerow(top_line)
    # write instances to file
    for index in progress_bar_file(instances):
        wtr.writerow(instances[index])


# ----------------------------------------------


# -------- build distance matrix --------
def calc_distances(p):
    row = []
    for q in range(0, p + 1, 1):
        row.append(0)
    for q in range(p + 1, n, 1):
        u, v = instances[p], instances[q]
        d = dist.euclidean(u, v)
        # d = np.linalg.norm(np.array(u) - np.array(v))  # about twice as fast as dist.euclidean
        row.append(d)
    return row


print('calculate distances')
progress_bar_random = ProgressBar()
pool = Parallel(n_jobs=-1, verbose=1)
results = pool(delayed(calc_distances)(p) for p in progress_bar_random(range(0, n, 1)))

graph = np.array(results)
print(graph)

# ----------------------------------------------

# -------- calculate Minimum Spanning Tree --------
print('calculate Minimum Spanning Tree')
#  noinspection PyTypeChecker
mst = minimum_spanning_tree(graph, overwrite=True).toarray()
print(mst)

#  -------- randomly generate labels --------
labels = []
for i in range(n):
    labels.append(randint(1, 2))

# -------- calculate complexity measure --------
boundary_length = 0

for s in range(0, n, 1):
    for t in range(s + 1, n, 1):  # take advantage of matrix symmetry
        if (mst[s][t] != 0) and (labels[s] != labels[t]):
            boundary_length += 1

complexity = boundary_length / n

print('Complexity: ', complexity)
csv_input = pd.read_csv('../assets/verify_data.csv')
csv_input['label'] = labels
csv_input.to_csv('../assets/verify_data_with_labels.csv', index=False)
