from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from random import randint, random
import csv
import scipy.spatial.distance as dist
from itertools import product

# set print options for large arrays
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)


# -------- Parameters --------
m = 5  # number of features
n = 10  # number of instances
b = 9 / n  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
l = [0, 1]  # manually define a set of possible labels
# -------- End Parameters --------


possible_labels = list(product(l, repeat=n))


#  assign labels and calculate class boundary
def get_labels(k):
    _labels = {}
    for x in range(0, n, 1):
        _labels.update({x: possible_labels[k][x]})
    return _labels


# calculate number of edges connecting points of opposite classes is counted and divided by the
# total number of connections. This ratio is taken as the measure of boundary length
def class_boundary(_minimum_spanning_tree, _labels):
    boundary_length = 0
    for s in range(0, n, 1):
        for t in range(s + 1, n, 1):
            if (_minimum_spanning_tree[s][t] != 0) and (_labels[s] != _labels[t]):
                boundary_length += 1
    return boundary_length / n


# create dataset without labeling
distribution_dict = {}
instances = {}

for j in range(0, m, 1):
    # set distribution for feature j
    mu, sigma = randint(0, 100), randint(0, 100)  # TODO distribution for mu and sigma itself ?
    # print('Mu: %d, Sigma: %s' % (mu, sigma))
    distribution_dict.update({j: [mu, sigma]})

for i in range(0, n, 1):
    instance = []
    for j in range(0, m, 1):
        instance.append(np.random.normal(distribution_dict[j][0], distribution_dict[j][1], 1)[0])
    instances.update({i: instance})

# store data in csv file
with open('../assets/data.csv', 'w') as f:
    wtr = csv.writer(f, delimiter=',')  # TODO Semicolon as delimiter
    for index in instances:
        wtr.writerow(instances[index])

# build graph
graph = []
for p in range(0, n, 1):
    row = []
    for q in range(0, p + 1, 1):
        row.append(0)
    for q in range(p + 1, n, 1):
        u, v = instances[p], instances[q]
        d = dist.euclidean(u, v)
        row.append(d)
    graph.append(row)

# convert to numpy array
graph = np.array(graph)
print(graph)
print()

# calculate Minimum Spanning Tree
# noinspection PyTypeChecker
mst = minimum_spanning_tree(graph).toarray()
print(mst)

num = 0
labels = get_labels(num)

# while class_boundary(mst, labels) != b:
while num < 1024:
    labels = get_labels(num)
    num += 1
    print(class_boundary(mst, labels))
    print(labels.values())
    print()

# print(class_boundary(mst, labels))
