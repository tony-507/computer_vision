import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import filters

def plane_sweep_ncc(im_l,im_r,start,steps,wid):
	""" Find disparity image using normalized cross-correlation. """

	m,n = im_l.shape[:2]

	# Arrays to hold different sums
	mean_l = np.zeros((m,n))
	mean_r = np.zeros((m,n))
	s = np.zeros((m,n))
	s_l = np.zeros((m,n))
	s_r = np.zeros((m,n))

	# Arrays to contain depth planes
	dmaps = np.zeros((m,n,steps))

	# Compute mean of patch
	filters.uniform_filter(im_l,wid,mean_l)
	filters.uniform_filter(im_r,wid,mean_r)

	# Normalized images
	norm_l = im_l - mean_l
	norm_r = im_r - mean_r

	# Try different disparities
	for displ in range(steps):
		# Move left image to right, compute sums
		filters.uniform_filter(np.roll(norm_l,-displ-start)*norm_r,wid,s) # sum nominator
		filters.uniform_filter(np.roll(norm_l,-displ-start)*np.roll(norm_l,-displ-start),wid,s_l)
		filters.uniform_filter(norm_r*norm_r,wid,s_r) # sum denominator

		# Store ncc scores
		dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

	# Return best depth at each pixel
	return np.argmax(dmaps,axis=2)

def plane_sweep_gaussian(im_l,im_r,start,steps,wid):
	""" Find disparity image using normalized cross-correlation with Gaussian weighted neighborhood. """

	m,n = im_l.shape[:2]

	# Arrays to hold different sums
	mean_l = np.zeros((m,n))
	mean_r = np.zeros((m,n))
	s = np.zeros((m,n))
	s_l = np.zeros((m,n))
	s_r = np.zeros((m,n))

	# Arrays to contain depth planes
	dmaps = np.zeros((m,n,steps))

	# Compute mean of patch
	filters.gaussian_filter(im_l,wid,0,mean_l)
	filters.gaussian_filter(im_r,wid,0,mean_r)

	# Normalized images
	norm_l = im_l - mean_l
	norm_r = im_r - mean_r

	# Try different disparities
	for displ in range(steps):
		# Move left image to right, compute sums
		filters.gaussian_filter(np.roll(norm_l,-displ-start)*norm_r,wid,0,s) # sum nominator
		filters.gaussian_filter(np.roll(norm_l,-displ-start)*np.roll(norm_l,-displ-start),wid,0,s_l)
		filters.gaussian_filter(norm_r*norm_r,wid,0,s_r) # sum denominator

		# Store ncc scores
		dmaps[:,:,displ] = s/np.sqrt(s_l*s_r)

	# Return best depth at each pixel
	return np.argmax(dmaps,axis=2)

def show_plot(im_l,im_r,start,steps,wid):

	plt.subplot(2,2,1)
	plt.gray()
	plt.imshow(im_l)

	plt.subplot(2,2,2)
	plt.gray()
	plt.imshow(im_r)

	plt.subplot(2,2,3)
	plt.imshow(plane_sweep_ncc(im_l,im_r,start,steps,wid))
	print('Done')

	plt.subplot(2,2,4)
	plt.imshow(plane_sweep_gaussian(im_l,im_r,start,steps,wid))
	print('Done')

	plt.show()

if __name__ == '__main__':
	im_l = np.array(Image.open('../../images/stereo/im2.png').convert('L'))
	im_r = np.array(Image.open('../../images/stereo/im6.png').convert('L'))

	steps = 100
	start = 4
	wid = 9

	show_plot(im_l,im_r,start,steps,wid)






