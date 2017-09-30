# TODO only create file with unlabeled and normally distributed instances ones

from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from random import randint, random
import csv
import scipy.spatial.distance as dist

# -------- Parameters --------
m = 5  # number of features
n = 10  # number of instances
b = 0.5  # the desired complexity, defined by the length of the class boundary, b ∈[0,1].
l = [0, 1]  # manually define a set of possible labels
# -------- End Parameters --------

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
    wtr = csv.writer(f, delimiter=';')  # Semicolon as delimiter
    for index in instances:
        wtr.writerow(instances[index])

# build graph
np.set_printoptions(threshold=np.inf, precision=2, linewidth=np.inf)
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

graph = np.array(graph)
print(graph)
print()
# noinspection PyTypeChecker
mst = minimum_spanning_tree(graph).toarray()
print(mst)

# assign labels and calculate class boundary
labels = {}
for i in range(0, n, 1):
    labels.update({i: randint(0, 1)})


# calculate number of edges connecting points of opposite classes is counted and divided by the
# total number of connections. This ratio is taken as the measure of boundary length
# If there are n vertices in the graph, then each spanning tree has n − 1 edges
# -> number of connection = n -1
def class_boundary(minimum_spanning_tree, labels):
    boundary_length = 0
    for s in range(0, n, 1):
        for t in range(s + 1, n, 1):
            if (minimum_spanning_tree[s][t] != 0) and (labels[s] != labels[t]):
                boundary_length += 1
    return boundary_length / (n - 1)

print(class_boundary(mst, labels))
