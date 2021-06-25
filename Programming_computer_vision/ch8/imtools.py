import numpy as np
import matplotlib.pyplot as plt


def plot_2D_boundary(plot_range,points,decisionfcn,labels,values=[0]):
	""" Plot_range is (xmin,xmax,ymin,ymax), points is a list
	of class points, decisionfcn is a funtion to evaluate,
	labels is a list of labels that decisionfcn returns for each class,
	values is a list of decision contours to show. """

	clist = ['b','r','g','k','m','y'] # colors for the classes

	# Evaluate on a grid and plot contour of decision function
	x = np.arange(plot_range[0],plot_range[1],.1)
	y = np.arange(plot_range[2],plot_range[3],.1)
	xx,yy = np.meshgrid(x,y)
	xxx,yyy = xx.flatten(),yy.flatten() # List of x,y in grid
	zz = np.array(decisionfcn(xxx,yyy))
	zz = zz.reshape(xx.shape)
	# Plot contour(s) at values
	plt.contour(xx,yy,zz,values)

	# For each class, plot the points with '*' for correct and 'o' for incorrect
	for i in range(len(points)):
		d = decisionfcn(points[i][:,0],points[i][:,1])
		correct_ndx = labels[i]==d
		incorrect_ndx= labels[i]!=d
		plt.plot(points[i][correct_ndx,0],points[i][correct_ndx,1],'*',color=clist[i])
		plt.plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],'o',color=clist[i])

	plt.axis('equal')
	plt.show()