# Scripts containing different filters
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Fill image with zero
def fill_im(img,r,c):
	m,n = img.shape
	out_img = np.zeros((m+r-1,n+c-1))
	out_img[r-1:,c-1:] = img
	return out_img

# Warp image to deal with boundary effect
def warp_im(img,r,c):
	r1,c1 = img.shape
	r -= 1
	c -= 1
	out_img = np.zeros((r1+r,c1+c))
	# Boundary condition: 1D filter
	# Assume no trivial filter
	if r == 0:
		out_img[:,:c] = img[:,-c:]
	elif c == 0:
		out_img[:r,:] = img[-r:,:]
	else:
		out_img[:r,:c] = img[-r:,-c:]
		out_img[:r,c:] = img[-r:,:]
		out_img[r:,:c] = img[:,-c:]
	out_img[r:,c:] = img
	return out_img

# Convolution function
def conv(img,ker):
	"""Convolute img with ker, change to float to avoid numerical error"""
	img = img.astype(float)
	r,c = ker.shape
	if len(img.shape) == 2:
		r1,c1 = img.shape
		d = 1
		img = img.reshape((r1,c1,1))
	elif len(img.shape) == 3:
		r1,c1,d = img.shape
	else:
		print('Invalid input format.')
		return

	warp_img = warp_im(img,r,c)
	for dp in range(d):
		for i in range(r1):
			for j in range(c1):
				img[i,j,dp] = np.sum(warp_img[i:i+r,j:j+c]*ker[-1::-1,-1::-1])

	if d == 1:
		img = img.reshape((r1,c1))
	return img

# Separable convolution function
def sep_conv(img,ker,tol=1):
	"""Apply convolution in separable manner, tol is used to involve more singular values"""
	U,S,V = np.linalg.svd(ker)
	out_img = img
	U = -U
	V = -V
	for i in range(tol):
		k = np.sqrt(S[i])
		out_img = conv(out_img,(k*U[:,i]).reshape((-1,1)))
		out_img = conv(out_img,(k*V[i,:]).reshape((1,-1)))
	return out_img

# Box filter function
def box(img,r,c):
	K = np.ones((r,c))/(r*c)
	out_img = conv(img,K)
	return out_img,K

# Compute distance matrix for Gaussian related tasks
def dist_mat(r,c):
	center = np.array([(r+1)/2-1,(c+1)/2-1])
	D = np.zeros((r,c))
	for i in range(r):
		for j in range(c):
			D[i,j] = (i-center[0])**2+(j-center[1])**2
	return D

# Gaussian filter
def gaussian(img,r=5,c=5,sigma=5):
	t = sigma**2
	K = np.exp(-dist_mat(r,c)/(2*t))
	K/=np.sum(K)
	out_img = conv(img,K)
	return out_img,K

# Laplacian of Gaussian
def LoG(img,r=5,c=5,sigma=5):
	t = sigma**2
	D = dist_mat(r,c)
	K = np.exp(-D/(2*t))*(D/(t**2)-2/t)
	K/=np.sum(K)
	out_img = conv(img,K)
	return out_img,K

# Compute summed area table
def compute_sum(img):
	img = img.astype(float)
	m,n = img.shape
	S = np.zeros((m,n))
	S[0,0] = img[0,0]
	# First row
	for j in range(1,n):
		S[0,j] = S[0,j-1] + img[0,j]
	# First column
	for i in range(1,m):
		S[i,0] = S[i-1,0] + img[i,0]
	for i in range(1,m):
		for j in range(1,n):
			S[i,j] = S[i-1,j] + S[i,j-1] - S[i-1,j-1] + img[i,j]
	return S

# Perform box filter with the table
def box_by_table(img,r,c,S=None):
	m,n = img.shape
	if S is None:
		S = compute_sum(img)
	S = fill_im(S,r,c)
	res = S[r-1:,c-1:] - S[:m,c-1:] - S[r-1:,:n] + S[:m,:n]
	return res/(r*c)

"""Nonlinear Filters"""

# Median filtering
def median(img,r=5,c=5):
	r1,c1 = img.shape
	img_extend = fill_im(img,r,c)
	for i in range(r1):
		for j in range(c1):
			img[i,j] = np.median(img_extend[i:i+r,j:j+c])
	return img

# Common morphology
def dilate(img,r=5,c=5):
	struct = np.ones((r,c))
	img = conv(img,struct)
	img = (img >= 1)*1
	return img.astype('uint8')

def erode(img,r=5,c=5):
	struct = np.ones((r,c))
	img = conv(img,struct)
	img = (img == r*c)*1
	return img.astype('uint8')

def majority(img,r=5,c=5):
	struct = np.ones((r,c))
	img = conv(img,struct)
	img = (img >= r*c//2)*1
	return img.astype('uint8')

def open(img,r=5,c=5):
	return dilate(erode(img,r,c),r,c)

def close(img,r=5,c=5):
	return erode(dilate(img,r,c),r,c)

# Manhattan Distance transform
def block_dist_transform(img):
	r,c = img.shape
	D = np.zeros(img.shape)
	D1 = np.zeros(img.shape)

	# Forward scan
	if img[0,0]>0:
		D1[0,0] = 10000
	for j in range(1,c):
		if img[0,j] != 0:
			D1[0,j] = 1+D1[0,j-1]
	for i in range(1,r):
		if img[i,0] != 0:
			D1[i,0] = 1+D1[i-1,0]
	for i in range(1,r):
		for j in range(1,c):
			if img[i,j] != 0:
				D1[i,j] = 1+min(D1[i-1,j],D1[i,j-1])
	            
	# Backward scan
	if img[r-1,c-1]>0:
		D[r-1,c-1] = 10000
	for j in range(c-2,-1,-1):
		if img[0,j] != 0:
			D[0,j] = min(1+D[0,j+1],D1[0,j])
	for i in range(r-2,-1,-1):
		if img[i,0] != 0:
			D[i,0] = min(1+D[i+1,0],D1[i,0])
	for i in range(r-2,-1,-1):
		for j in range(c-2,-1,-1):
			if img[i,j] != 0:
				D[i,j] = min(1+D[i+1,j],1+D[i,j+1],D1[i,j])
	            
	return D