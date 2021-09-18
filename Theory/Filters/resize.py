import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image

def resize(img,dim_new,r,c,ker='bilinear'):
	"""Resize an image with output shape dim_new with given kernel of size (r,c)"""
	if len(img.shape) == 2:
		w,h = img.shape
		d = 1
		img = img.reshape((w,h,1))
	elif len(img.shape) == 3:
		w,h,d = img.shape
	else:
		print('Error: Invalid input format.')
		return

	img = img.astype(float)

	f, bd = choose_kernel(ker)

	r1 = dim_new[0]/w
	r2 = dim_new[1]/h
	res = np.zeros((dim_new[0],dim_new[1],d))
	count = 0

	print(f'Sampling rate: ({r1},{r2})')

	for dp in range(d):
		for i in range(dim_new[0]):
			# Separable kernel execution
			# Compute h(i-rk) by:
			# 1) find suitable k
			# 2) Form a vector of the form [h(i-rk_0),h(i-rk_1),...,h(i-rk_n)]^T
			bound_i = np.maximum(np.array([(i-bd[1])/r1,(i-bd[0])/r1]).astype(int),0)
			ker_i = f(i-r1*np.arange(bound_i[0],bound_i[1]+1)).reshape((-1,1))
			ker_i = ker_i/np.sum(ker_i)
			for j in range(dim_new[1]):
				# Similarly, we compute h(j-rl)
				bound_j = np.maximum(np.array([(j-bd[1])/r2,(j-bd[0])/r2]).astype(int),0)
				ker_j = f(j-r2*np.arange(bound_j[0],bound_j[1]+1)).reshape((-1,1))
				ker_j = ker_j/np.sum(ker_j)

				patch = img[bound_i[0]:bound_i[1]+1,bound_j[0]:bound_j[1]+1,dp]
				res[i,j,dp] = patch*(ker_i@ker_j.T)

	if d==1:
		res = res.reshape(dim_new)

	return res.astype('uint8')

def choose_kernel(ker):
	# Choose kernel, vectorised for better performance
	if ker == 'spline':
		f1 = lambda x: 1-2*x**2+abs(x**3) if abs(x)<1 else 0
		f2 = lambda x: -(abs(x)-1)*(abs(x)-2)**2 if abs(x)>1 and abs(x)<2 else 0
		f = lambda x: f1(x)+f2(x)
		bd = [-2,2]
	else:
		f = lambda x: max(0,1-abs(x))
		bd = [-1,1]

	return np.vectorize(f), bd

if __name__ == '__main__':
	img = np.array(Image.open('../../images/beatles.png').convert('L'))
	patch = img[200:301,200:301]
	t0 = time()
	img2 = resize(patch,(150,150),5,5)
	print(f'{time()-t0}s used')

	plt.figure()
	plt.gray()
	plt.imshow(patch)

	# plt.figue()
	# plt.gray()
	# plt.imshow(img2)
	# plt.show()