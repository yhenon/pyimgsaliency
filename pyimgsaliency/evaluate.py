import os
import pdb
import pyimgsaliency
import cv2
import pdb
import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
def evaluate(img_dir,gt_dir,methods):

	valid_extensions = ['.jpg','.png']

	results_precision = {}
	results_recall = {}
	for filename in os.listdir(img_dir):
		if filename.endswith(".jpg") or filename.endswith(".png"): 
			basename = os.path.splitext(filename)[0]
			gt_image_path = None
			if os.path.isfile(gt_dir + '/' + basename + '.png'):
				gt_image_path = gt_dir + '/' + basename + '.png'
			elif os.path.isfile(gt_dir + '/' + basename + '.jpg'):
				gt_image_path = gt_dir + '/' + basename + '.jpg'
			else:
				print('No match in gt directory for file' + str(filename) + ', skipping.')
				continue
			print(img_dir + '/' + filename)
			sal_image = pyimgsaliency.get_saliency_mbd(img_dir + '/' + filename).astype('uint8')
			gt_image = cv2.imread(gt_image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
			cv2.imshow('sal',sal_image)
			cv2.imshow('img',gt_image)
			cv2.waitKey(0)
			if gt_image.shape != sal_image.shape:
				print('Size of image and GT image does not match, skipping')
				continue
			#precision, recall, thresholds = sklearn.metrics.precision_recall(y_true,y_scores)
			'''
			precisions = {}
			recalls = {}
			
			num_pixels = sal_image.shape[0] * sal_image.shape[1]
			
			p = np.count_nonzero(gt_image)
			n = num_pixels - p
			
			for v in xrange(0,255):
				culled = np.copy(sal_image)
				culled[culled < v] = 0
				if np.count_nonzero(culled) == 0:
					recall = 1
					sensitivity = 0
				else:
					tp = np.count_nonzero(culled & gt_image)
					recall = float(tp)/p
					precision = float(tp) / np.count_nonzero(culled)

				precisions[v] = precision
				recalls[v] = recall

			results_precision[filename] = precisions
			results_recall[filename] = recalls

			x = []
			y = []
			for x1 in precisions:
				#print x1
				x.append(precisions[x1])
				y.append(recalls[x1])
			plt.plot(x,y)
			plt.show()
			'''
			#pdb.set_trace()
	return (results_precision,results_recall)
