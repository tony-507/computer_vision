import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import *

class spectral_cluster(object):

	def __init__(self,data):
		self.data = data
		self.n_data = data.shape[0]
		self.n_feature = data.shape[1]

	def cluster(self,k):
		"""Code for spectral clustering"""

		# Compute distance matrix
		S = np.array([[np.sqrt(np.sum((data[i]-data[j])**2)) for i in range(self.n_data)] for j in range(self.n_data)],'f')

		# Create Laplcaian matrix
		rowsum = np.sum(S,axis=0)
		D = np.diag(1/np.sqrt(rowsum))
		I = np.identity(self.n_data)
		L = I - D@S@D

		# Compute eigenvectors of L
		U,S,V = np.linalg.svd(L)

		self.k = k
		# Create feature vectors from self.k first eigenvectors
		# by stacking eigenvectors as columns
		self.features = np.array(V[:self.k]).T
		self.centroids, self.distortion = kmeans(self.features,self.k)
		self.code, self.distance = vq(self.features,self.centroids)

	def plot_cluster(self):
		for c in range(self.k):
			ind = np.where(self.code==c)[0]
			plt.figure()
			for i in range(np.minimum(len(ind),39)):
				plt.subplot(4,10,i+1)
				plt.imshow(data[i].reshape((20,20)))
				plt.axis('equal')
				plt.axis('off')

		plt.show()

if __name__ == '__main__':
	data = np.loadtxt('arial_data.txt')
	cluster_model = spectral_cluster(data)

	cluster_model.cluster(10)
	cluster_model.plot_cluster()