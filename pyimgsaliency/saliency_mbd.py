import math
import sys
import operator
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
import scipy.signal
import skimage
import skimage.io
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize

import pdb

def raster_scan(img,L,U,D):
	n_rows = len(img)
	n_cols = len(img[0])

	for x in xrange(1,n_rows - 1):
		for y in xrange(1,n_cols - 1):
			ix = img[x][y]
			d = D[x][y]

			u1 = U[x-1][y]
			l1 = L[x-1][y]

			u2 = U[x][y-1]
			l2 = L[x][y-1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x][y] = b1
				U[x][y] = max(u1,ix)
				L[x][y] = min(l1,ix)
			else:
				D[x][y] = b2
				U[x][y] = max(u2,ix)
				L[x][y] = min(l2,ix)

	return True

def raster_scan_inv(img,L,U,D):
	n_rows = len(img)
	n_cols = len(img[0])

	for x in xrange(n_rows - 2,1,-1):
		for y in xrange(n_cols - 2,1,-1):

			ix = img[x][y]
			d = D[x][y]

			u1 = U[x+1][y]
			l1 = L[x+1][y]

			u2 = U[x][y+1]
			l2 = L[x][y+1]

			b1 = max(u1,ix) - min(l1,ix)
			b2 = max(u2,ix) - min(l2,ix)

			if d <= b1 and d <= b2:
				continue
			elif b1 < d and b1 <= b2:
				D[x][y] = b1
				U[x][y] = max(u1,ix)
				L[x][y] = min(l1,ix)
			else:
				D[x][y] = b2
				U[x][y] = max(u2,ix)
				L[x][y] = min(l2,ix)
	return True

def mbd(img, num_iters):

	if len(img.shape) != 2:
		print('did not get 2d np array to fast mbd')
		return None
	if (img.shape[0] <= 3 or img.shape[1] <= 3):
		print('image is too small')
		return None

	L = np.copy(img)
	U = np.copy(img)
	D = float('Inf') * np.ones(img.shape)
	D[0,:] = 0
	D[-1,:] = 0
	D[:,0] = 0
	D[:,-1] = 0

	# unfortunately, iterating over numpy arrays is very slow
	img_list = img.tolist()
	L_list = L.tolist()
	U_list = U.tolist()
	D_list = D.tolist()

	for x in xrange(0,num_iters):
		if x%2 == 1:
			raster_scan(img_list,L_list,U_list,D_list)
		else:
			raster_scan_inv(img_list,L_list,U_list,D_list)

	return np.array(D_list)


def get_saliency_mbd(img_path):

	# Saliency map calculation based on: Minimum Barrier Salient Object Detection at 80 FPS

	img = skimage.io.imread(img_path)
	img_mean = np.mean(img,axis=(2))
	sal = mbd(img_mean,3)

	# get the background map

	# paper uses 30px for an image of size 300px, so we use 10%
	(n_rows,n_cols,n_channels) = img.shape
	img_size = math.sqrt(n_rows * n_cols)
	border_thickness = int(math.floor(0.1 * img_size))

	img_lab = img_as_float(skimage.color.rgb2lab(img))
	
	px_left = img_lab[0:border_thickness,:,:]
	px_right = img_lab[n_rows - border_thickness-1:-1,:,:]

	px_top = img_lab[:,0:border_thickness,:]
	px_bottom = img_lab[:,n_cols - border_thickness-1:-1,:]
	
	px_mean_left = np.mean(px_left,axis=(0,1))
	px_mean_right = np.mean(px_right,axis=(0,1))
	px_mean_top = np.mean(px_top,axis=(0,1))
	px_mean_bottom = np.mean(px_bottom,axis=(0,1))


	px_left = px_left.reshape((n_cols*border_thickness,3))
	px_right = px_right.reshape((n_cols*border_thickness,3))

	px_top = px_top.reshape((n_rows*border_thickness,3))
	px_bottom = px_bottom.reshape((n_rows*border_thickness,3))

	cov_left = np.cov(px_left.T)
	cov_right = np.cov(px_right.T)

	cov_top = np.cov(px_top.T)
	cov_bottom = np.cov(px_bottom.T)

	cov_left = np.linalg.inv(cov_left)
	cov_right = np.linalg.inv(cov_right)
	cov_top = np.linalg.inv(cov_top)
	cov_bottom = np.linalg.inv(cov_bottom)


	u_left = np.zeros(sal.shape)
	u_right = np.zeros(sal.shape)
	u_top = np.zeros(sal.shape)
	u_bottom = np.zeros(sal.shape)

	u_final = np.zeros(sal.shape)

	print('cov ...')

	for x in xrange(0,u_left.shape[0]):
		for y in xrange(0,u_left.shape[1]):
			v = img_lab[x,y,:]
			v_diff_left = v - px_mean_left
			v_left = np.dot(v_diff_left,cov_left)
			v_left = np.dot(v_left,v_diff_left)
			u_left[x,y] = math.sqrt(v_left)

			v_diff_right = v - px_mean_right
			v_right = np.dot(v_diff_right,cov_right)
			v_right = np.dot(v_right,v_diff_right)
			u_right[x,y] = math.sqrt(v_right)

			v_diff_top = v - px_mean_top
			v_top = np.dot(v_diff_top,cov_top)
			v_top = np.dot(v_top,v_diff_top)
			u_top[x,y] = math.sqrt(v_top)

			v_diff_bottom = v - px_mean_bottom
			v_bottom = np.dot(v_diff_bottom,cov_bottom)
			v_bottom = np.dot(v_bottom,v_diff_bottom)
			u_bottom[x,y] = math.sqrt(v_bottom)

	max_u_left = np.max(u_left)
	max_u_right = np.max(u_right)
	max_u_top = np.max(u_top)
	max_u_bottom = np.max(u_bottom)

	u_left = u_left / max_u_left
	u_right = u_right / max_u_right
	u_top = u_top / max_u_top
	u_bottom = u_bottom / max_u_bottom

	u_max = np.maximum(np.maximum(np.maximum(u_left,u_right),u_top),u_bottom)

	u_final = (u_left + u_right + u_top + u_bottom) - u_max

	u_max_final = np.max(u_final)
	sal_max = np.max(sal)
	sal = sal / sal_max + u_final / u_max_final

	#pdb.set_trace()

	return sal

