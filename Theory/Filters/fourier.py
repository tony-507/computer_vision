import numpy as np
import matplotlib.pyplot as plt

def plot2D(func,r,c,mid=False):
	"""Display an image with given dimension (r,c)"""
	I = np.zeros((r,c))
	if mid == True:
		r0 = -r//2
		r1 = r//2+1
		c0 = -c//2
		c1 = c//2+1
	else:
		r0 = 0
		c0 = 0
		r1 = r
		c1 = c
	for i in range(r0,r1):
		for j in range(c0,c1):
			I[i,j] = func(i,j)

	# Normalize
	Imax = np.amax(I)
	Imin = np.amin(I)
	I = (I-Imin)/(Imax-Imin)*255

	# Plot
	plt.gray()
	plt.imshow(I.astype('uint8'))
	plt.show()


def plot3D(func,r,c,x=50):
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	# Make data.
	X = np.arange(-r/2, r/2, r/x)
	Y = np.arange(-c/2, c/2, c/x)
	l = X.size
	X, Y = np.meshgrid(X, Y)
	R = np.zeros((l,l))
	for i in range(l):
	    for j in range(l):
	        R[i,j] = func(i,j)

	# Plot the surface.
	surf = ax.plot_surface(X, Y, R, linewidth=0, antialiased=False)

	plt.show()

class Fourier:
	def transform2D(self,img):
		"""2D Fourier transform"""
		r,c = img.shape
		rweight = ((np.arange(r)).reshape((-1,1)))*(np.ones(c).T)
		cweight = (np.ones(r).reshape((-1,1)))*(np.arange(c)).reshape((1,-1))
		self.real2D = lambda kx,ky: np.sum(img*np.cos(2*np.pi*(rweight*kx+cweight*ky)/(r*c)))
		self.imag2D = lambda kx,ky: np.sum(-img*np.sin(2*np.pi*(rweight*kx+cweight*ky)/(r*c)))
		


class DCT:
	def __init__(self,img):
		self.r,self.c = img.shape[:2]
		rweight = ((np.arange(self.r) + 0.5)*np.pi/self.r).reshape((-1,1))
		cweight = ((np.arange(self.c) + 0.5)*np.pi/self.c).reshape((-1,1))
		self.compute = lambda kx,ky: np.sum(np.cos(rweight*kx)*np.cos(cweight*ky).T*img)