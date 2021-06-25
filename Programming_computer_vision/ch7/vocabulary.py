import numpy as np
import matplotlib.pyplot as plt
import sift
import pickle
from scipy.cluster.vq import *

class Vocabulary(object):

	def __init__(self,name):
		self.name = name
		self.voc = []
		self.idf = []
		self.trainingdata = []
		self.nbr_words = 0

	def train(self,featfile,k=100,subsampling=10):
		""" Train a vocabulary from features in files listed
		in featfile using k-means with k number of words.
		Subsampling of training data can be used for speedup. """

		nbr_images = len(featfile)

		print('Reading features...',end='',flush=True)
		# Generate features using SIFT
		descr = []
		descr.append(sift.read_features_from_file(featfile[0])[1])
		descriptors = descr[0]
		for i in range(1,nbr_images):
			descr.append(sift.read_features_from_file(featfile[i])[1])
			descriptors = np.vstack((descriptors,descr[i]))
		print('Done')

		# k-means: last number determines the number of runs
		print('Begin clustering...',end='',flush=True)
		self.voc,distortion = kmeans(descriptors[::subsampling,:],k,1)
		self.nbr_words = self.voc.shape[0]
		print('Done')
		
		# Go through all training images and project on vocabulary
		imwords = np.zeros((nbr_images,self.nbr_words))
		for i in range(nbr_images):
			imwords[i] = self.project(descriptors[i].reshape(1,-1))

		nbr_occurences = np.sum( (imwords > 0)*1,axis=0)

		self.idf = np.log((1.0*nbr_images)/(1.0*nbr_occurences+1))
		self.trainingdata = featfile


	def project(self,descriptors):
		""" Project descriptors on the vocabulary to create a histogram of words. """

		# Histogram of image words
		imhist = np.zeros((self.nbr_words))
		words,distance = vq(descriptors,self.voc)
		for w in words:
			imhist[w] += 1

		return imhist

if __name__ == '__main__':
	nbr_images = 1000
	featfile = []
	for i in range(10):
		tmp = 'ukbench0000'+str(i)+'.sift'
		featfile.append(tmp)
	for i in range(10,100):
		tmp = 'ukbench000'+str(i)+'.sift'
		featfile.append(tmp)
	# for i in range(100,1000):
	# 	tmp = 'ukbench00'+str(i)+'.sift'
	# 	featfile.append(tmp)

	voc = Vocabulary('ukbenchtest')
	voc.train(featfile,100,10)

	with open('vocabulary_1.pkl','wb') as f:
		pickle.dump(voc,f)
	print(f'Vocabulary is {voc.name} with {voc.nbr_words} words')







