# import pandas as pd
# import numpy as np
# import time
#
# mst_edges = [[0, 1], [0, 2], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [1, 0], [1, 2], [1, 5],
#              [1, 6], [1, 7], [1, 8]]
#
#
# # fitness function
# def evaluate(individual_):
#     # number of edges connecting points of opposite classes is counted and divided by the
#     # total number of connections. This ratio is taken as the measure of boundary length
#     boundary_length = 0
#     for edge in mst_edges:
#         if individual_[edge[0]] != individual_[edge[1]]:
#             boundary_length += 1
#     return boundary_length / n
#
#
# n = 10
# individual = np.random.randint(2, size=n)
#
# print(individual)
# print(evaluate(individual))
#
# # print('Time:', (time.time()-start))

print([5.0]*3)