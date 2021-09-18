import numpy as np
import matplotlib.pyplot as plt
from time import time
from filters import block_dist_transform

class Components(object):

	def __init__(self,img,d=None):
		self.img = img
		self.compo = np.zeros(img.shape)
		self.n_component = 1
		if d is None:
			self.d = lambda x,y: np.sum((x-y)**2)
		else:
			self.d = d

	def horizontal_run(self):
		"""Horizontal run over the image"""
		r,c = self.img.shape
		self.n_component = 1
		for i in range(r-1):
			# First column
			# Not yet assigned
			if self.compo[i,0] == 0:
			    self.compo[i,0] = self.n_component
			    self.n_component += 1
			# Vertical Node
			if self.img[i+1,0] == self.img[i,0]:
				self.compo[i+1,0] = self.compo[i,0]
			for j in range(1,c):
				# Both pixel has same values
				if self.compo[i,j] == 0 and self.img[i,j-1] == self.img[i,j]:
					self.compo[i,j] = self.compo[i,j-1]
				# Left neighbour has different value
				elif self.compo[i,j] == 0:
					self.compo[i,j] = self.n_component
					self.n_component += 1
				# For already assigned pixel, need only check vertical node
				if self.img[i+1,j] == self.img[i,j]:
					self.compo[i+1,j] = self.compo[i,j]

		# Last row
		# First entry
		if self.compo[r-1,0] == 0:
			self.compo[r-1,0] = self.n_component
			self.n_component += 1
		# Remaining entries
		for j in range(1,c):
			# Same component
			if self.compo[r-1,j] == 0 and self.img[r-1,j-1] == self.img[r-1,j]:
				self.compo[r-1,j] = self.compo[r-1,j-1]
			# Different value
			elif self.compo[r-1,j] == 0:
				self.compo[r-1,j] = self.n_component
				self.n_component += 1

	def merge(self):
		"""Merge components from horizontal run"""
		r,c = self.compo.shape
		self.n_component = 1
		# Backward checking
		for i in range(r):
			for j in range(c):
				if i != r-1 and self.img[i+1,j] == self.img[i,j] and self.compo[i+1,j] != self.compo[i,j]:
					self.compo[self.compo==self.compo[i+1,j]] = self.compo[i,j]
				if j != 0 and self.img[i,j-1] == self.img[i,j] and self.compo[i,j-1] != self.compo[i,j]:
					self.compo[self.compo==self.compo[i,j-1]] = self.compo[i,j]
		# Rearrange indices
		for i in range(1,int(np.amax(self.compo))+1):
			if self.compo[self.compo==i].size > 0:
				self.compo[self.compo==i] = self.n_component
				self.n_component += 1

	def find_components(self):
		t0 = time()
		print("Horizontal run...",end='',flush=True)
		self.horizontal_run()
		print(f"{time()-t0}s")
		t0 = time()
		print("Merging...",end='',flush=True)
		self.merge()
		print(f"{time()-t0}s")

	def plot_components(self,cmap='brg'):
		plt.imshow(self.compo, cmap=cmap, vmin=1, vmax=self.n_component)
		plt.show()

	def area(self,n):
		return self.compo[self.compo==n].size

	def perimeter(self,n):
		"""Use Manhattan distance transform to find perimeter efficiently"""
		rmax = np.amax(np.where(self.compo==n)[0])
		rmin = np.amin(np.where(self.compo==n)[0])
		cmax = np.amax(np.where(self.compo==n)[1])
		cmin = np.amin(np.where(self.compo==n)[1])
		bin_map = np.zeros((rmax-rmin+1,cmax-cmin+1))

		bin_transformed = block_dist_transform(bin_map)
		return bin_transformed[bin_transformed==1].size

	def compute_centroid(self,n):
		ndx = np.where(self.compo==n)
		return np.mean(ndx,axis=1)

	def moment(n):
		ndx = np.where(self.compo==n)
		centroid = np.mean(ndx,axis=1)
		a = np.sum((ndx[0]-centroid[0])**2)
		b = np.sum((ndx[0]-centroid[0])*(ndx[1]-centroid[1]))
		c = np.sum((ndx[1]-centroid[1])**2)
		return np.array([[a,b],[b,c]])


