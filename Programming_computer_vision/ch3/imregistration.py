import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy import linalg
from xml.dom import minidom
from scipy import ndimage
import imageio
import os

def read_points_from_xml(xmlFileName):
	"""Read control points from xml file"""

	xmldoc = minidom.parse(xmlFileName)
	facelist = xmldoc.getElementsByTagName('face')
	faces = {}
	for xmlFace in facelist:
		filename = xmlFace.attributes['file'].value
		xf = int(xmlFace.attributes['xf'].value)
		yf = int(xmlFace.attributes['yf'].value)
		xs = int(xmlFace.attributes['xs'].value)
		ys = int(xmlFace.attributes['ys'].value)
		xm = int(xmlFace.attributes['xm'].value)
		ym = int(xmlFace.attributes['ym'].value)
		faces[filename] = np.array([xf,yf,xs,ys,xm,ym])

	return faces

def compute_rigid_transform(refpoints,points):
	"""Compute rotation, scale and translation for aligning refpoints to points"""

	A = np.array([[points[0],-points[1],1,0],\
		[points[1],points[0],1,0],\
		[points[2],-points[3],1,0],\
		[points[3],points[2],1,0],\
		[points[4],-points[5],1,0],\
		[points[5],points[4],1,0]])

	y = np.array([refpoints[0],\
		refpoints[1],\
		refpoints[2],\
		refpoints[3],\
		refpoints[4],\
		refpoints[5]])

	# Least square solution
	a,b,tx,ty = linalg.lstsq(A,y)[0]
	R = np.array([[a,-b],[b,a]]) # Rotation matrix including scale

	return R,tx,ty

def rigid_alignment(faces,path,plotflag=False):
	""" Align images rigidly and save as new images.
	path determines where the aligned images are saved
	set plotflag=True to plot the images. """

	# Take the points in first face as refpoints
	refpoints = list(faces.values())[0]

	# Warp each image using affine transform
	for face in faces:
		points = faces[face]
		R,tx,ty = compute_rigid_transform(refpoints,points)
		T = np.array([[R[1][1],R[1][0]],[R[0][1],R[0][0]]])

		im = np.array(Image.open(os.path.join(path,face)))
		im2 = np.zeros(im.shape,'uint8')

		# Warp each color channel
		for i in range(len(im.shape)):
			im2[:,:,i] = ndimage.affine_transform(im[:,:,i],linalg.inv(T),offset=[-ty,-tx])

		if plotflag:
			plt.imshow(im2)
			plt.show()

		# Crop border
		h,w = im2.shape[:2]
		border = int((w+h)/20)

		imageio.imwrite(os.path.join(path, 'aligned/'+face),im2[border:h-border,border:w-border,:])







