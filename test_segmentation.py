import math
import sys

import skimage
import skimage.io

import networkx as nx

import matplotlib.pyplot as plt
import numpy as np

import scipy.spatial.distance

import pdb

from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.optimize import minimize

def cost_function(x):
	cost = 0
	for i in range(0,len(x)):
		cost += w_bg[i] * (x[i]*x[i])
		cost += wCtr[i] * (x[i] - 1) * (x[i] - 1)
	print cost
	return cost

def path_length(path,weight_type):
	dist = 0.0
	for i in range(1,len(path)):
		dist += G[path[i - 1]][path[i]][weight_type]
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
	
img_path = sys.argv[1]

img = skimage.io.imread(img_path)

img_lab = img_as_float(skimage.color.rgb2lab(img))
#img_lab = img_as_float(img)

img_rgb = img_as_float(img)
img_gray = img_as_float(skimage.color.rgb2gray(img))

segments_slic = slic(img_rgb, n_segments=250, compactness=10, sigma=1)

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
	colors[v] = np.mean(img_lab[grid==v])
	#img_gray[grid == v] = colors[v]
	x_pix = gridx[grid == v]
	y_pix = gridy[grid == v]
	if np.any(x_pix == 0) or np.any(y_pix == 0) or np.any(x_pix == ncols - 1) or np.any(y_pix == nrows - 1):
		boundary[v] = 1
	else:
		boundary[v] = 0

G = nx.Graph()

for edge in edges:
	pt1 = edge[0]
	pt2 = edge[1]
	euclidean_distance = scipy.spatial.distance.euclidean(centers[pt1],centers[pt2]) / max_dist
	color_distance = scipy.spatial.distance.euclidean(colors[pt1],colors[pt2])
	G.add_edge(pt1, pt2, weight_color=color_distance, weight_spatial=euclidean_distance )

geodesic = np.zeros((len(vertices),len(vertices)),dtype=float)
spatial = np.zeros((len(vertices),len(vertices)),dtype=float)

all_shortest_paths_color = nx.shortest_path(G,source=None,target=None,weight='weight_color')

for v1 in vertices:
	for v2 in vertices:
		if v1 == v2:
			geodesic[v1,v2] = 0
			spatial[v1,v2] = 0
		else:
			geodesic[v1,v2] = path_length(all_shortest_paths_color[v1][v2],'weight_color')
			spatial[v1,v2] = scipy.spatial.distance.euclidean(centers[v1],centers[v2]) / max_dist

sigma_clr = 10.0
sigma_bndcon = 1.0
sigma_spa = 0.25

area = dict()
len_bnd = dict()
bnd_con = dict()
w_bg = dict()
ctr = dict()
wCtr = dict()

#for v1 in vertices:

for v1 in vertices:
	area[v1] = 0
	len_bnd[v1] = 0
	ctr[v1] = 0
	for v2 in vertices:
		d_app = geodesic[v1,v2]
		d_spa = spatial[v1,v2]
		w_spa = math.exp(- ((d_spa)*(d_spa))/(2.0*sigma_spa*sigma_spa))
		S = math.exp(- pow(d_app,2)/(2.0 * sigma_clr *sigma_clr))
		area[v1] += S
		len_bnd[v1] += S * boundary[v2]
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

img_disp1 = img_gray.copy()
img_disp2 = img_gray.copy()

x0 = 0.5 * np.ones(num_segments)
print('Optimising ... ')
res = minimize(cost_function, x0, method='BFGS',options={'xtol': 1e-8, 'disp': True})
print x0
for v in vertices:
	img_disp1[grid == v] = res.x[v]
	img_disp2[grid == v] = ctr[v]


plt.figure(1)
plt.subplot(211)
plt.imshow(img_disp1)
plt.subplot(212)
plt.imshow(img_disp2)

plt.show()

