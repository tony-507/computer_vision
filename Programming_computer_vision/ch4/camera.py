import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import linalg

class Camera(object):
	"""Class for representing pin-hole camera"""

	def __init__(self,P):
		"""Initialize P = K[R|t] camera model"""
		self.P = P
		self.K = None # Calibration matrix
		self.R = None # Rotation
		self.t = None # Translation
		self.c = None # Camera center

	def project(self,X):
		"""Project points in X (4*n array) and normalize coordinates"""

		x = self.P@X
		for i in range(3):
			x[i] /= x[2]
		return x

	def factor(self):
		""" Factorize the camera matrix into K,R,t as P = K[R|t] by QR factorization"""

		# Factor first 3*3 part
		K,R = linalg.rq(self.P[:,:3])

		# Make diagonal of K positive
		T = np.sign(np.diag(np.sign(np.diag(K))))

		self.K = K@T
		self.R = T@R
		self.t = linalg.inv(self.K)@self.P[:,3]

		return self.K,self.R,self.t

	def center(self):
		""" Compute and return the camera center. """

		if self.c is not None:
			return self.c
		else:
			# Compute c by factoring
			self.factor()
			self.c = -self.R.T@self.t
			return self.c

def rotation_matrix(a):
	""" Creates a 3D rotation matrix for rotation around the axis of the vector a. """

	R = np.eye(4)
	R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
	return R

def my_calibration(sz):
	"""Calibrate a camera in landscape view using similar triangle. Change the resolution if necessary.

	For the bible_calib.jpg, dY = 11.2 cm, dX = 16.5cm, dZ = 75cm, dy = 535, dx = 778"""
	# fx = 3536, fy = 3583
	row, col = sz
	fx = 3536*col/4032
	fy = 3583*row/3024
	K = np.diag([fx,fy,1])
	K[0,2] = 0.5*col
	K[1,2] = 0.5*row

	return K

























