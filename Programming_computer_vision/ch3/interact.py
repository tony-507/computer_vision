import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import warp
import homography
import imregistration

def view_image():
	path = input('Please enter the filename of the image: ')
	path = '../../images/' + path

	im = np.array(Image.open(path))

	plt.figure()
	if len(im.shape) == 2:
		plt.gray()
	plt.imshow(im)
	plt.show()

def write_landmarks():
	"""Write n manual landmarks to a txt file"""
	filename = input('Please enter the filename of the image: ')
	path = '../../images/' + filename
	n = int(input('How many landmarks? '))

	im = np.array(Image.open(path).convert('L'))

	print(f'Please select {n} landmarks')

	plt.figure()
	plt.gray()
	plt.imshow(im)
	# plt.show()

	landmarks = plt.ginput(n)
	landmarks = np.roll(landmarks,1,axis=0)

	np.savetxt(path[:-4] + '_landmark.txt',landmarks)

def affine_map():
	# path1 = input('Please enter the file name of the first image: ')
	# path1 = '../../images/' + path1
	path1 = '../../images/beatles.png'

	fromim = np.array(Image.open(path1).convert('L'))

	# path2 = input('Please enter the file name of the second image: ')
	# path2 = '../../images/' + path2
	path2 = '../../images/billboard.jpg'

	toim = np.array(Image.open(path2).convert('L'))

	# Process first image to create mesh
	x,y = np.meshgrid(range(3),range(3))
	x = (fromim.shape[1]/2) * x.flatten()
	y = (fromim.shape[0]/2) * y.flatten()

	# triangulate
	tri = warp.triangulate_points(x,y)
	tri -= 1

	# Retrieve landmarks of second image
	tp = np.loadtxt(path2[:-4] + '_landmark.txt')

	# convert points to hom. coordinates
	fp = np.vstack((y,x,np.ones((1,len(x))))).astype(int)
	tp = np.vstack((tp[:,1],tp[:,0],np.ones((1,len(tp))))).astype(int)

	# Apply piecewise affine map
	im = warp.pw_affine(fromim,toim,fp,tp,tri)

	# Plot
	plt.figure()
	plt.gray()
	plt.imshow(im)
	plt.show()

def im_registration():
	xmlFileName = input('Please input the name of the XML file: ')
	xmlFileName = '../../images/' + xmlFileName
	points = imregistration.read_points_from_xml(xmlFileName)

	imregistration.rigid_alignment(points,'../../images/')

if __name__ == '__main__':
	# Require the two images
	print('This is a command line interface for ch3')

	while True:
		print('Main Menu')
		print('-------------------------')
		print('0: Instructions')
		print('1: Displaying Image')
		print('2: Writing Landmarks')
		print('3: Image Blending')
		print('4: Image Registration')
		print('L: Exit Program')

		opt = input('Please choose a function...')

		if opt == '0':
			print('This interface contains functions described in chapter 3 of the book.')
			print('Displaying Image: Simply display selected image.')
			print('Writing landmarks: Create a txt containing manually labelled landmarks.')
			print('Image Blending: Apply affine map on first image to blend to second image.')
			print('Image Registration: Register a collection of images according to a given XML file')
		elif opt == '1':
			view_image()
		elif opt == '2':
			write_landmarks()
		elif opt == '3':
			affine_map()
		elif opt == '4':
			im_registration()
		elif opt == 'L':
			break
		else:
			print('Invalid. Please try again.')