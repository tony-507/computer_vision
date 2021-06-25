import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_fundamental(x1,x2):
	""" Computes the fundamental matrix from corresponding points
	(x1,x2 3*n arrays) using the normalized 8 point algorithm.
	Each row is constructed as
	[x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

	n = x1.shape[1]

	if x2.shape[1] != n:
		raise ValueError("Number of points don't match.")

	# Build matrix for equations
	A = np.zeros((n,9))
	for i in range(n):
		A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
		x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
		x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

	# Compute linear least square solution
	U,S,V = np.linalg.svd(A)
	F = V[-1].reshape(3,3)

	# Ensure F is rank 2 by making last eigenvalue of F zero
	U,S,V = np.linalg.svd(F)
	S[2] = 0
	F = U@np.diag(S)@V

	return F

def compute_epipole(F):
	""" Computes the (right) epipole from a fundamental matrix F.
	(Use with F.T for left epipole.) """

	# Return null space of F
	U,S,V = np.linalg.svd(F)
	e = V[-1]

	return e/e[2]

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
	""" Plot the epipole and epipolar line F*x=0 in an image.
	F is the fundamental matrix and x a point in the other image."""

	m,n = im.shape[:2]
	line = F@x

	# epipolar line parameter and values
	t = np.linspace(0,n,100)
	lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

	# take only line points inside the image
	ndx = (lt>=0) & (lt<m)
	plt.plot(t[ndx],lt[ndx],linewidth=2)

	if show_epipole:
		if epipole is None:
			epipole = compute_epipole(F)
		plt.plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

def triangulate_point(x1,x2,P1,P2):
	"""Point-pair triangulation from SVD"""

	M = np.zeros((6,6))
	M[:3,:4] = P1
	M[3:,:4] = P2
	M[:3,4] = -x1
	M[3:,5] = -x2

	U,S,V = np.linalg.svd(M)
	X = V[-1,:4]

	return X/X[3]

def triangulate(x1,x2,P1,P2):
	""" Two-view triangulation of points in x1,x2 (3*n homog. coordinates). """

	n = x1.shape[1]
	if x2.shape[1] != n:
		raise ValueError("Number of points don't match")

	X = [triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
	return np.array(X).T

def compute_P(x,X):
	""" Compute camera matrix from pairs of
	2D-3D correspondences (in homog. coordinates). """

	n = x.shape[1]
	if X.shape[1] != n:
		raise ValueError("Number of points don't match")

	# Create coefficient matrix
	M = np.zeros((3*n,n+12))
	for i in range(n):
		M[3*i,0:4] = X[:,i]
		M[3*i+1,4:8] = X[:,i]
		M[3*i+2,8:12] = X[:,i]
		M[3*i:3*i+3,i+12] = -x[:,i]

	U,S,V = np.linalg.svd(M)

	return V[-1,:12].reshape((3,4))

def skew(a):
	""" Skew matrix A such that a x v = Av for any v. """

	return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_fundamental(F):
	""" Computes the second camera matrix (assuming P1 = [I 0]) from a fundamental matrix. """

	e = compute_epipole(F.T) # Left epipole
	Te = skew(e)
	return np.vstack((Te@F.T,e)).T

def compute_P_from_essential(E):
	""" Computes the second camera matrix (assuming P1 = [I 0])
	from an essential matrix.

	Output is a list of four possible camera matrices. """

	# Make sure E is rank 2
	U,S,V = np.linalg.svd(E)
	if np.det(U@V) < 0:
		V = -V
	E = U@np.diag([1,1,0])@V

	# Create matrices
	Z = skew([0,0,-1])
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

	# Return all four possible solutions
	P2 = [np.vstack((np.dot(U,np.dot(W,V)).T,U[:,2])).T,
		np.vstack((np.dot(U,np.dot(W,V)).T,-U[:,2])).T,
		np.vstack((np.dot(U,np.dot(W.T,V)).T,U[:,2])).T,
		np.vstack((np.dot(U,np.dot(W.T,V)).T,-U[:,2])).T]

	return P2








