# Import necessary packages
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Function to create panorama from two images
def create_panorama(img1,img2):
	"""Create a panorama joining im1 and im2, where im1 is at the left, im2 is at the right

	Input: img1 (array), img2 (array)
	Output: img_output (array)"""

	# Detection by SIFT
	sift = cv.SIFT_create()

	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	# Set parameters
	MIN_MATCH_COUNT = 10
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	flann = cv.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1,des2,k=2)

	# Store good matches by Lowe's ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	# Find homography if we have enough matches, otherewise show a message
	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		h1,w1 = img1.shape
		h2,w2 = img2.shape
		pts = np.float32([ [0,0],[0,h2],[w2,h2],[w2,0] ]).reshape(-1,1,2)
		dst = cv.perspectiveTransform(pts,M)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None

	# Image stitching
	pt_list = np.concatenate((pts,dst),axis=0)

	[x_min, y_min] = np.int32(pt_list.min(axis=0).ravel() - 0.5) # Select minimum corner
	[x_max, y_max] = np.int32(pt_list.max(axis=0).ravel() + 0.5)

	translate_dist = [-x_min,-y_min]

	H_translation = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]])

	# Warp img1
	img_output = cv.warpPerspective(img1,H_translation@M,(x_max-x_min,y_max-y_min))

	# Append img2
	img_output[-y_min:h2-y_min,-x_min:w2-x_min] = img2

	return img_output

if __name__ == '__main__':
	print('This program allows creation of panorama.')
	print('To begin, please save the source image at images/panorama and named them as tag0.jpg, tag1.jpg,...')
	print('Local directory: '+os.getcwd())
	path = input('Please input the path to find the images: ')
	tag = input('Please input the tag for the images: ')
	n_image = int(input('Please input the number of images: '))

	imlist = {} # dict storing image array

	print('Reading images...',end='')
	for i in range(n_image):
		imlist[i] = cv.imread('images/panorama/'+tag+str(i)+'.jpg',0)
	print('Done')

	img_output = imlist[0]

	print('Creating panorama',end='')
	for i in range(1,n_image):
		print('.',end='')
		img_output = create_panorama(img_output,imlist[i])
	print('Done')

	cv.imwrite('images/panorama/'+tag+'_result.jpg',img_output)