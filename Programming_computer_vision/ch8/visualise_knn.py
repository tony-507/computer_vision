import numpy as np
import matplotlib.pyplot as plt
import knn
import imtools

def gen_shift_data(n=200):
	"""Generate data with linear decision boundary"""

	n = 200 # Number of data

	# Two normal distributions
	class1 = 0.6*np.random.randn(n,2)
	class2 = 1.2*np.random.randn(n,2) + np.array([5,1])
	labels = np.hstack((np.ones(n),-np.ones(n)))

	return class1,class2,labels

def gen_circ_data():
	"""Generate data with circular decision boundary"""

	n = 200 # Number of data

	#One on line and one on circle
	class1 = 0.6*np.random.randn(n,2)
	r = 0.8*np.random.randn(n,1) + 5
	angle = 2*np.pi*np.random.randn(n,1)
	class2 = np.hstack((r*np.cos(angle),r*np.sin(angle)))
	labels = np.hstack((np.ones(n),-np.ones(n)))

	return class1,class2,labels

class1, class2, labels = gen_circ_data()
model = knn.KnnClassifier(labels,np.vstack((class1,class2)))
# Generate test data
class1, class2, labels = gen_circ_data()

def classify(x,y,model = model):
	"""Helper function for visualisation"""

	return np.array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])



imtools.plot_2D_boundary([-6,6,-6,6],[class1,class2],classify,[1,-1])