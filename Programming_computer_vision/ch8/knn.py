import numpy as np

class KnnClassifier(object):

	def __init__(self,labels,samples):
		""" Initialize classifier with training data. """

		self.labels = labels
		self.samples = samples

	def classify(self,point,k=3):
		""" Classify a point against k nearest in the training data, return label. """

		# Compute distance to all training points
		dist = np.array([L2dist(point,s) for s in self.samples])

		# Sort
		ndx = dist.argsort()

		# Use dictionary to store the k nearest
		votes = {}
		for i in range(k):
			label = self.labels[ndx[i]]
			votes.setdefault(label,0)
			votes[label] += 1

		return max(votes)




def L2dist(p1,p2):
	return np.sqrt(np.sum((p1-p2)**2))