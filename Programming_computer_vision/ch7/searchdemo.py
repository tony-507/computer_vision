import cherrypy, os, urllib, pickle
import imagesearch
import random
from vocabulary import Vocabulary

class SearchDemo(object):

	def __init__(self):
		# Load list of images
		with open('webimlist.txt') as f:
			self.imlist = f.readlines()

		self.nbr_images = len(self.imlist)
		self.ndx = list(range(self.nbr_images))

		# Load vocabulary
		with open('vocabulary_1.pkl','rb') as f:
			self.voc = pickle.load(f)

		# Set max number of results to show
		self.maxres = 15

		# Absolute path to this file
		self.abs_path = "/Users/chantony/Desktop/study/it/python/imaging_pillow/Programming_computer_vision/ch7/"

		# Header and footer html
		self.header = """
		<!doctype html>
		<head>
		<title>Image search example</title> </head>
		<body>
		"""
		self.footer = """
		</body>
		</html>
		"""

	def index(self,query=None):
		self.src = imagesearch.Searcher('test.db',self.voc)

		html = self.header
		html += """
		<br />
		Click an image to search. <a href='?query='>Random selection</a> of images. <br /><br />
		"""

		if query:
			# query the database and get top images

			res = self.src.query(query)[:self.maxres]
			for dist,ndx in res:
				imname = self.src.get_filename(ndx)
				html += "<a href='?query="+imname+"'>"
				html += "<img src='ukbench/full/"+imname+"' width='200' />"
				html += "</a>"
		else:
			# Show random selection if no query
			random.shuffle(self.ndx)
			for i in self.ndx[:self.maxres]:
				imname = self.imlist[i]
				html += "<a href='?query="+imname+"'>"
				html += "<img src='ukbench/full/"+imname+"' width='200' />"
				html += "</a>"

		html += self.footer
		return html

	index.exposed = True

cherrypy.quickstart(SearchDemo(),'/',config=os.path.join(os.path.dirname(__file__), 'service.conf'))