import sift
import pickle
import imagesearch
from vocabulary import Vocabulary

if __name__ == '__main__':

	featfile = []
	imlist = []
	for i in range(10):
		tmp = 'ukbench0000'+str(i)
		imlist.append(tmp+'.jpg')
		featfile.append(tmp+'.sift')
	for i in range(10,100):
		tmp = 'ukbench000'+str(i)
		imlist.append(tmp+'.jpg')
		featfile.append(tmp+'.sift')

	nbr_images = len(featfile)

	print('Loading dictionary...',end='',flush=True)
	# Load dictionary
	with open('vocabulary_1.pkl','rb') as f:
		voc = pickle.load(f)
	print('Done')


	"""Writing to database"""
	# print('Adding to database...')
	# # Create indexer
	# indx = imagesearch.Indexer('test.db',voc)
	# indx.create_tables()

	# # Go through all images, project features on vocabulary and insert
	# for i in range(nbr_images)[:100]:
	# 	locs,descr = sift.read_features_from_file(featfile[i])
	# 	indx.add_to_index(imlist[i],descr)
	# print('Done')


	# # commit to database
	# indx.db_commit()

	"""Querying an image"""
	src = imagesearch.Searcher('test.db',voc)
	locs,descr = sift.read_features_from_file(featfile[0])
	iw = voc.project(descr)

	print('Average matching score:')
	print(imagesearch.compute_ukbench_score(src,imlist))

	res = [w[1] for w in src.query(imlist[0])[:6]]
	imagesearch.plot_results(src,res)
