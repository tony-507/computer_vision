import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_imlist(path):
	"""Return a list of filenames for jpg images in a directory"""
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im,sz):
	"""Resize an image array using PIL"""
	pil_im = Image.fromarray(uint8(im))
	return np.array(pil_im.resize(sz))

def histeq(im,n_bins = 256):
	"""Histogram equalization of a grayscale image"""
	im_hist, bins, patches = plt.hist(im.flatten(), n_bins, density=True)
	cdf = im_hist.cumsum()
	cdf = 255*cdf/cdf[-1] # Normalize

	# Map im by cdf
	im2 = np.interp(im.flatten(),bins[:-1],cdf)

	return im2.reshape(im.shape), cdf

def compute_average(imlist):
	"""Compute the average of a list of images"""

	# Open the first image
	im_average = np.array(Image.open(imlist[0]),'f')

	for i_name in imlist[1:]:
		try:
			im_average += np.array(Image.open(i_name))
		except:
			print('%s ... skipped' % i_name)

	im_average /= len(imlist)

	return np.array(im_average,'uint8')

def pca(X):
	"""Principal Component Analysis"""

	n_data, dim = X.shape

	mu_X = X.mean(axis=0)
	X = X - mu_X

	if dim > n_data:
		# Comp
		M = X@X.T # Covariance matrix
		e, EV = linalg.eigh(M)
		tmp = X.T@EV
		V = tmp[::-1] # Reverse the order
		S = np.sqrt(e)[::-1]
		for i in range(V.shape[1]):
			V[:,i] /= S
	else:
		# SVD used
		U,S,V = linalg.svd(X)
		V = V[:n_data]

	return V,S,mean_X


























