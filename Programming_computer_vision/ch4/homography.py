import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize(points):
	"""Normalize a collection of points in homogenous coordinates so that the last row = 1"""

	for row in points:
		row /= points[-1]

	return points

def make_homog(points):
	"""Convert a set of points (dim*n) to homogeneous coordinates"""

	return np.vstack((points,np.ones((1,points.shape[1]))))

def H_from_points(fp,tp):
	"""Find H such that tp = H * fp using linear DLT method. Points are conditioned automatically"""

	if fp.shape != tp.shape:
		raise RuntimeError('Number of points do not match')

	# Condition points due to numerical reasons

	# --from points--
	m = np.mean(fp[:2],axis=1)
	maxstd = np.max(np.std(fp[:2],axis=1)) + 1e-9
	C1 = np.diag([1/maxstd,1/maxstd,1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp = C1@fp

	# --to points--
	m = np.mean(tp[:2],axis=1)
	maxstd = np.max(np.std(tp[:2],axis=1)) + 1e-9
	C2 = np.diag([1/maxstd,1/maxstd,1])
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp = C2@tp

	# Create the corresponding coefficient matrix
	n_corr = fp.shape[1]
	A = np.zeros((2*n_corr,9))
	for i in range(n_corr):
		A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,fp[0][i]*tp[0][i],fp[1][i]*tp[0][i],tp[0][i]]
		A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,fp[0][i]*tp[1][i],fp[1][i]*tp[1][i],tp[1][i]]

	U,S,V = np.linalg.svd(A)
	H = V[8].reshape((3,3))

	# Decondition
	H = np.linalg.inv(C2)@H@C1

	# Normalize and return
	return H/H[2,2]

def Haffine_from_points(fp,tp):
	"""Find affine H such that tp = H * fp"""

	if fp.shape != tp.shape:
		raise RuntimeError('number of points do not match')

	# Condition points

	# --from points--
	m = np.mean(fp[:2],axis=1)
	maxstd = np.max(np.std(fp[:2],axis=1)) + 1e-9
	C1 = np.diag([1/maxstd,1/maxstd,1])
	C1[0][2] = -m[0]/maxstd
	C1[1][2] = -m[1]/maxstd
	fp = C1@fp

	# --to points--
	m = np.mean(tp[:2],axis=1)
	C2 = C1.copy()
	C2[0][2] = -m[0]/maxstd
	C2[1][2] = -m[1]/maxstd
	tp = C2@tp

	# Conditioned point have zero mean, so translation is zero
	A = np.concatenate((fp[:2],tp[:2]),axis=0)
	U,S,V = np.linalg.svd(A.T)

	# Create B and C matrices
	tmp = V[:2].T
	B = tmp[:2]
	C = tmp[2:4]

	tmp2 = np.concatenate((C@np.linalg.pinv(B),np.zeros((2,1))), axis=1)
	H = np.vstack((tmp2,[0,0,1]))

	# Decondition
	H = np.linalg.inv(C2)@H@C1

	return H/H[2,2]












