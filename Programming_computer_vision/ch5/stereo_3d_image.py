import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import sfm
import camera
from mpl_toolkits.mplot3d import axes3d

# Read images
img1 = cv.imread('../../images/stereo/fan0.jpg',0)
img2 = cv.imread('../../images/stereo/fan1.jpg',0)

# Feature detection by SIFT
sift = cv.SIFT_create()

kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

# Feature matching by FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Make inlier points homogeneous

pts1 = np.vstack((pts1.T,np.ones(pts1.shape[0])))
pts2 = np.vstack((pts2.T,np.ones(pts2.shape[0])))

cam1 = camera.Camera(np.hstack((np.diag([1,1,1]),np.zeros((3,1)))))
K = camera.my_calibration((4032,3024))

# Compute P by assuming a form [S_eF|e]

P_fund = sfm.compute_P_from_fundamental(F)
cam_fund = camera.Camera(P_fund)

points_3d_fund = sfm.triangulate(pts1,pts2,cam1.P,cam_fund.P)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(-points_3d_fund[0],points_3d_fund[1],points_3d_fund[2],'k.')

plt.show()