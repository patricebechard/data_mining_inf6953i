
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from collections import Counter

from operator import itemgetter
import sys



data = np.genfromtxt('data.csv', delimiter=',')



def norm_data(data):

	means = np.mean(data, axis=0)
	std_devs = np.std(data, axis=0)

	return (data - means) / std_devs

def return_outliers(data, n_clusters=4, mixture=False, visualize_data=False):

	if mixture:
		clusterizer = GaussianMixture(n_components=n_clusters).fit(data)
		labels = clusterizer.predict(data)
	else:
		clusterizer = KMeans(n_clusters=n_clusters).fit(data)
		labels = clusterizer.labels_

	cntr = Counter(labels)

	kept_classes = []
	for elem in cntr.most_common():

		if elem[-1] <= 2:
			kept_classes.append(elem[0])

	outliers = []
	for cl in kept_classes:
		for i in range(len(data)):

			if labels[i] == cl:

				outliers.append(i)

	if visualize_data:
		show_data(data, labels, mixture)

	return outliers


def show_data(data, cluster_assignments, mixture=False):

	projection = PCA(n_components=2).fit_transform(data)

	plt.scatter(projection[:,0], projection[:,1], c=cluster_assignments)
	if mixture:
		plt.savefig('visualization_gmm.png')
	else:
		plt.savefig('visualization_kmeans.png')

if __name__ == "__main__":

	data = np.genfromtxt('data.csv', delimiter=',')
	data = norm_data(data)

	# return outliers for a number of configs
	for mixture in [False, True]:
		if mixture:
			print("Method : Gaussian Mixture Model")
		else:
			print("Method : K-Means")

		for n_clusters in range(4, 12):
			outliers = return_outliers(data, n_clusters, mixture=mixture)

			print("N clusters : %d ---- Outliers : %s"%(n_clusters, str(outliers)))


	# visualize the data
	outliers = return_outliers(data, n_clusters=8, visualize_data=True)

	outliers = return_outliers(data, n_clusters=8, mixture=True, visualize_data=True)

