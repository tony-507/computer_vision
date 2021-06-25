import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.spatial import Delaunay

import homography

def alpha_for_triangle(points,m,n):
	"""Create alpha map of size (m,n) for a triangle with corners defined by points"""

	alpha = np.zeros((m,n))
	for i in range(np.min(points[0]),np.max(points[0])):
		for j in range(np.min(points[1]),np.max(points[1])):
			x = np.linalg.solve(points,[i,j,1])
			if np.min(x) > 0: # Must be convex combination in this case
				alpha[i,j] = 1
	return alpha

def triangulate_points(x,y):
	""" Delaunay triangulation of 2D points. """

	tri = Delaunay(np.c_[x,y]).simplices
	return tri

def image_in_image(im1,im2,tp):
	"""Put im1 in im2 with an affine transformation
	such that the corners are as close to tp as possible.
	tp are homogeneous and counterclockwise from top left"""

	# Points to warp from
	m,n = im1.shape[:2]
	fp = np.array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

	# First triangle
	tp1 = tp[:,:3]
	fp1 = fp[:,:3]

	# Compute H
	H = homography.Haffine_from_points(tp1,fp1)
	im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])

	# alpha for triangle
	alpha = alpha_for_triangle(tp1,im2.shape[0],im2.shape[1])
	im3 = (1-alpha)*im2 + alpha*im1_t

	# Second triangle
	tp2 = tp[:,[0,2,3]]
	fp2 = fp[:,[0,2,3]]

	# Compute H
	H = homography.Haffine_from_points(tp2,fp2)
	im1_t = ndimage.affine_transform(im1,H[:2,:2],(H[0,2],H[1,2]),im2.shape[:2])

	# alpha for triangle
	alpha = alpha_for_triangle(tp2,im2.shape[0],im2.shape[1])
	im4 = (1-alpha)*im3 + alpha*im1_t

	return im4


def pw_affine(fromim,toim,fp,tp,tri):
	""" Warp triangular patches from an image.
	fromim = image to warp
	toim = destination image
	fp = from points in hom. coordinates
	tp = to points in hom. coordinates
	tri = triangulation. """

	im = toim.copy()

	# Check if image is grayscale or colored
	is_color = len(fromim.shape) == 3

	# Create image to warp to (needed if iterate colors)
	im_t = np.zeros(im.shape,'uint8')

	for t in tri:
		# Compute affine transform
		H = homography.Haffine_from_points(tp[:,t],fp[:,t])

		if is_color:
			for col in range(fromim.shape[2]):
				im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
		else:
			im_t = ndimage.affine_transform(fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])

		# alpha for triangle
		alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])

		# Add triangle to image
		im[alpha>0] = im_t[alpha>0]

	return im






