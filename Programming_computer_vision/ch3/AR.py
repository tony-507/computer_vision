import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
import pygame, pygame.image
from pygame.locals import *
from OpenGL.GLUT import *# Teapot example

def set_projection_from_camera(K):
	"""Set view from a camera matrix"""

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()

	fx = K[0,0]
	fy = K[1,1]
	fovy = 2*np.arctan(0.5*height/fy)*180/np.pi
	aspect = (width*fy)/(height*fx)

	# Define near and far clipping planes
	near = 0.1
	far = 100.0

	# Set perspective
	gluPerspective(fovy,aspect,near,far)
	glViewport(0,0,width,height)

def set_modelview_from_camera(Rt):
	"""Set the model view matrix from camera pose"""

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

	# rotate object 90 deg around x-axis so that z-axis is up
	Rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])

	# Set rotation to best approximation
	R = Rt[:,:3]
	U,S,V = np.linalg.svd(R)
	R = U@V
	R[0,:] = -R[0,:] # Change sign of x-axis

	# Set translation
	t = Rt[:,3]

	# Set up 4*4 model view matrix
	M = np.eye(4)
	M[:3,:3] = R@Rx
	M[:3,3] = t

	# Transpose and flatten to get column order
	M = M.T
	m = M.flatten()

	# Replace model view with the new matrix
	glLoadMatrixf(m)

def draw_background(imname):
	"""Draw background image using a quad"""

	# Load background image (should be .bmp) to OpenGL texture
	bg_image = pygame.image.load(imname).convert()
	bg_data = pygame.image.tostring(bg_image,"RGBX",1)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	# Bind the texture
	glEnable(GL_TEXTURE_2D)
	glBindTexture(GL_TEXTURE_2D,glGenTextures(1))
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,bg_data)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)

	# create quad to fill the whole window
	glBegin(GL_QUADS)
	glTexCoord2f(0.0,0.0); glVertex3f(-1.0,-1.0,-1.0)
	glTexCoord2f(1.0,0.0); glVertex3f( 1.0,-1.0,-1.0)
	glTexCoord2f(1.0,1.0); glVertex3f( 1.0, 1.0,-1.0)
	glTexCoord2f(0.0,1.0); glVertex3f(-1.0, 1.0,-1.0)
	glEnd()

	# clear the texture
	glDeleteTextures(1)

def draw_teapot(size):
	""" Draw a red teapot at the origin. """

	glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_DEPTH_TEST)
	glClear(GL_DEPTH_BUFFER_BIT)

	# draw red teapot
	glMaterialfv(GL_FRONT,GL_AMBIENT,[0,0,0,0])
	glMaterialfv(GL_FRONT,GL_DIFFUSE,[0.5,0.0,0.0,0.0])
	glMaterialfv(GL_FRONT,GL_SPECULAR,[0.7,0.6,0.6,0.0])
	glMaterialf(GL_FRONT,GL_SHININESS,0.25*128.0)
	glutSolidTeapot(size)

def setup():
	"""Set up window and pygame environment"""

	pygame.init()
	pygame.display.set_mode((width,height),OPENGL | DOUBLEBUF)
	pygame.display.set_caption('OpenGL AR demo')

if __name__ == '__main__':
	width, height = 4032, 3024
	K = np.array([[3536,0,2016],[0,3583,1512],[0,0,1]])
	Rt = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

	setup()
	draw_background('../../images/pose/bible_front.jpg')
	#set_projection_from_camera(K)
	#set_modelview_from_camera(Rt)
	#draw_teapot(0.02)

	while True:
		event = pygame.event.poll()
		if event.type in (QUIT,KEYDOWN):
			break 
		pygame.display.flip()




























