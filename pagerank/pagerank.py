import numpy as np 
import copy

def pagerank(graph, n_iter, rank0, d):

	rank = rank0
	N = len(graph)
	degree = np.sum(graph, axis=-1)

	if n_iter < 0 :
		epsilon = 0.001
		prev_rank = np.ones(N) # fake data for the first iteration
		while np.sum(np.abs(rank-prev_rank)) > epsilon:
			prev_rank = copy.deepcopy(rank)
			rank = (1-d) / N + d * np.sum(rank*graph.T / degree, axis=1)

	else:
		for i in range(n_iter):
			rank = (1-d) / N + d * np.sum(rank*graph.T / degree, axis=1)

	return rank

if __name__ == "__main__":

	d = 0.6
	n_iter = -1

	graph = np.array([[0, 1, 1, 0],
		              [0, 0, 1, 0],
		              [0, 0, 0, 1],
		              [0, 0, 1, 0]])

	rank0 = np.ones(len(graph)) / len(graph)

	rank = pagerank(graph, n_iter=n_iter, rank0=rank0, d=d)

	print(rank)