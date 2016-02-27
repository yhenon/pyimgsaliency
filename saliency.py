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

def S(x1,x2,geodesic,sigma_clr=10):
	return math.exp(-pow(geodesic[x1,x2],2)/(2*sigma_clr*sigma_clr))

def cost_function(x,adjacency,smoothness,w_bg,wCtr):
	cost = 0.0
	for i in xrange(0,len(x)):
		cost += w_bg[i] * x[i] * x[i]
		cost += wCtr[i] * (x[i] - 1) * (x[i] - 1)
		for j in xrange(0,len(x)):
			cost+= adjacency[i,j] * smoothness[i,j] * (x[i] - x[j])*(x[i] - x[j])
	print cost
	return cost

def diff_cost_function(x,adjacency,smoothness,w_bg,wCtr):
	diff = np.zeros(len(x))
	for i in xrange(0,len(x)):
		diff[i] += 2 * w_bg[i] * x[i]
		diff[i] += 2 * wCtr[i] * (x[i] - 1)
		for j in xrange(0,len(x)):
			diff[i] += 2 * adjacency[i,j] * smoothness[i,j] * (x[i] - x[j])
			diff[j] += 2 * adjacency[i,j] * smoothness[i,j] * (x[j] - x[i])
	return diff


def path_length(path,G):
	dist = 0.0
	for i in xrange(1,len(path)):
		dist += G[path[i - 1]][path[i]]['weight']
	return dist

def make_graph(grid):
	# get unique labels
	vertices = np.unique(grid)
 
	# map unique labels to [1,...,num_labels]
	reverse_dict = dict(zip(vertices,np.arange(len(vertices))))
	grid = np.array([reverse_dict[x] for x in grid.flat]).reshape(grid.shape)
   
	# create edges
	down = np.c_[grid[:-1, :].ravel(), grid[1:, :].ravel()]
	right = np.c_[grid[:, :-1].ravel(), grid[:, 1:].ravel()]
	all_edges = np.vstack([right, down])
	all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1], :]
	all_edges = np.sort(all_edges,axis=1)
	num_vertices = len(vertices)
	edge_hash = all_edges[:,0] + num_vertices * all_edges[:, 1]
	# find unique connections
	edges = np.unique(edge_hash)
	# undo hashing
	edges = [[vertices[x%num_vertices],
			  vertices[x/num_vertices]] for x in edges] 
 
	return vertices, edges
	

def get_saliency_rbd(img_path):

	# Saliency map calculation based on:
	# Saliency Optimization from Robust Background Detection, Wangjiang Zhu, Shuang Liang, Yichen Wei and Jian Sun, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014

	img = skimage.io.imread(img_path)

	img_lab = img_as_float(skimage.color.rgb2lab(img))

	img_rgb = img_as_float(img)

	img_gray = img_as_float(skimage.color.rgb2gray(img))

	segments_slic = slic(img_rgb, n_segments=250, compactness=10, sigma=1, enforce_connectivity=True)

	num_segments = len(np.unique(segments_slic))

	nrows, ncols = segments_slic.shape
	max_dist = math.sqrt(nrows*nrows + ncols*ncols)
	print("Slic number of segments: %d" % num_segments)

	grid = segments_slic

	(vertices,edges) = make_graph(grid)

	gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

	centers = dict()
	colors = dict()
	distances = dict()
	boundary = dict()

	for v in vertices:
		centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
		colors[v] = np.mean(img_lab[grid==v],axis=0)

		x_pix = gridx[grid == v]
		y_pix = gridy[grid == v]

		if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == nrows - 1) or np.any(y_pix == ncols - 1):
			boundary[v] = 1
		else:
			boundary[v] = 0

	G = nx.Graph()

	#buid the graph
	for edge in edges:
		pt1 = edge[0]
		pt2 = edge[1]
		color_distance = scipy.spatial.distance.euclidean(colors[pt1],colors[pt2])
		G.add_edge(pt1, pt2, weight=color_distance )

	#add a new edge in graph if edges are both on boundary
	for v1 in vertices:
		if boundary[v1] == 1:
			for v2 in vertices:
				if boundary[v2] == 1:
					color_distance = scipy.spatial.distance.euclidean(colors[v1],colors[v2])
					G.add_edge(v1,v2,weight=color_distance)

	geodesic = np.zeros((len(vertices),len(vertices)),dtype=float)
	spatial = np.zeros((len(vertices),len(vertices)),dtype=float)
	smoothness = np.zeros((len(vertices),len(vertices)),dtype=float)
	adjacency = np.zeros((len(vertices),len(vertices)),dtype=float)

	sigma_clr = 10.0
	sigma_bndcon = 1.0
	sigma_spa = 0.25
	mu = 0.1

	all_shortest_paths_color = nx.shortest_path(G,source=None,target=None,weight='weight')

	for v1 in vertices:
		for v2 in vertices:
			if v1 == v2:
				geodesic[v1,v2] = 0
				spatial[v1,v2] = 0
				smoothness[v1,v2] = 0
			else:
				geodesic[v1,v2] = path_length(all_shortest_paths_color[v1][v2],G)
				spatial[v1,v2] = scipy.spatial.distance.euclidean(centers[v1],centers[v2]) / max_dist
				smoothness[v1,v2] = math.exp( - (geodesic[v1,v2] * geodesic[v1,v2])/(2.0*sigma_clr*sigma_clr)) + mu 

	for edge in edges:
		pt1 = edge[0]
		pt2 = edge[1]
		adjacency[pt1,pt2] = 1
		adjacency[pt2,pt1] = 1

	area = dict()
	len_bnd = dict()
	bnd_con = dict()
	w_bg = dict()
	ctr = dict()
	wCtr = dict()

	for v1 in vertices:
		area[v1] = 0
		len_bnd[v1] = 0
		ctr[v1] = 0
		for v2 in vertices:
			d_app = geodesic[v1,v2]
			d_spa = spatial[v1,v2]
			w_spa = math.exp(- ((d_spa)*(d_spa))/(2.0*sigma_spa*sigma_spa))
			area_i = S(v1,v2,geodesic)
			area[v1] += area_i
			len_bnd[v1] += area_i * boundary[v2]
			ctr[v1] += d_app * w_spa
		bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])
		w_bg[v1] = 1.0 - math.exp(- (bnd_con[v1]*bnd_con[v1])/(2*sigma_bndcon*sigma_bndcon))

	for v1 in vertices:
		wCtr[v1] = 0
		for v2 in vertices:
			d_app = geodesic[v1,v2]
			d_spa = spatial[v1,v2]
			w_spa = math.exp(- (d_spa*d_spa)/(2.0*sigma_spa*sigma_spa))
			wCtr[v1] += d_app * w_spa *  w_bg[v2]

	# normalise value for wCtr

	min_value = min(wCtr.values())
	max_value = max(wCtr.values())

	minVal = [key for key, value in wCtr.iteritems() if value == min_value]
	maxVal = [key for key, value in wCtr.iteritems() if value == max_value]

	for v in vertices:
		wCtr[v] = (wCtr[v] - min_value)/(max_value - min_value)

	img_disp1 = img_gray.copy()
	img_disp2 = img_gray.copy()

	print('Optimising ... ')
	x0 = 0.5 * np.ones(num_segments)
	res = minimize(cost_function, x0, method='BFGS',jac=diff_cost_function,args=(adjacency,smoothness,w_bg,wCtr),tol=10,options={'gtol': 1e-2, 'disp': True, 'maxiter': 10})

	for v in vertices:
		img_disp1[grid == v] = res.x[v]

	img_disp2 = img_disp1.copy()
	img_disp = np.zeros((img_disp1.shape[0],img_disp1.shape[1],3))
	for i in np.arange(0.5,1.0,0.05):
		img_disp2[img_disp1 < i] = 0
		img_disp2[img_disp1 >= i] = 1

		img_disp[:,:,0] = img_disp2
		img_disp[:,:,1] = img_disp2
		img_disp[:,:,2] = img_disp2



	return img_disp1

def get_saliency_ft(img_path):

	# Saliency map calculation based on:

	img = skimage.io.imread(img_path)

	img_rgb = img_as_float(img)

	img_lab = img_rgb 

	mean_val = np.mean(img_rgb,axis=(0,1))

	kernel_h = (1.0/16.0) * np.array([[1,4,6,4,1]])
	kernel_w = kernel_h.transpose()

	blurred_l = scipy.signal.convolve2d(img_lab[:,:,0],kernel_h,mode='same')
	blurred_a = scipy.signal.convolve2d(img_lab[:,:,1],kernel_h,mode='same')
	blurred_b = scipy.signal.convolve2d(img_lab[:,:,2],kernel_h,mode='same')

	blurred_l = scipy.signal.convolve2d(blurred_l,kernel_w,mode='same')
	blurred_a = scipy.signal.convolve2d(blurred_a,kernel_w,mode='same')
	blurred_b = scipy.signal.convolve2d(blurred_b,kernel_w,mode='same')

	im_blurred = np.dstack([blurred_l,blurred_a,blurred_b])

	sal = np.linalg.norm(mean_val - im_blurred,axis = 2)

	return sal

def binarise_saliency_map(saliency_map,method='clustering',threshold=0.5):

	if len(saliency_map.shape) != 2:
		print('Saliency map must be 2D')
		return 0

	if method == 'fixed':
		return (saliency_map > threshold)

	elif method == 'adaptive':
		adaptive_threshold = 2.0 * saliency_map.mean()
		print adaptive_threshold
		return (saliency_map > adaptive_threshold)

	elif method == 'clustering':
		print('Not yet implemented')
		return 0

	else:
		print("Method must be one of fixed, adaptive or clustering")
		return 0
