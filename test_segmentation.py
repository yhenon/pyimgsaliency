import skimage
import networkx as nx

import math
import sys

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import scipy.spatial.distance

import pdb


def path_length(path):
	dist = 0.0
	for i in range(1,len(path)):
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
	
img_path = sys.argv[1]

img = skimage.io.imread(img_path)

img_lab = img_as_float(skimage.color.rgb2lab(img))
img_rgb = img_as_float(img)
img_gray = img_as_float(skimage.color.rgb2gray(img))

segments_slic = slic(img_rgb, n_segments=250, compactness=10, sigma=1)

num_segments = len(np.unique(segments_slic))

nrows, ncols = segments_slic.shape

print("Slic number of segments: %d" % num_segments)

grid = segments_slic

(vertices,edges) = make_graph(grid)

print vertices

gridx, gridy = np.mgrid[:grid.shape[0], :grid.shape[1]]

centers = dict()
colors = dict()
distances = dict()
boundary = dict()

for v in vertices:
	centers[v] = [gridy[grid == v].mean(), gridx[grid == v].mean()]
	colors[v] = np.mean(img_lab[grid==v])
	img_gray[grid == v] = colors[v]
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
	euclidean_distance = scipy.spatial.distance.euclidean(centers[pt1],centers[pt2])
	color_distance = scipy.spatial.distance.euclidean(colors[pt1],colors[pt2])
	G.add_edge(pt1, pt2, weight=color_distance )

geodesic = np.zeros((len(vertices),len(vertices)),dtype=float)

all_shortest_paths = nx.shortest_path(G,source=None,target=None,weight='weight')
for v1 in vertices:
	for v2 in vertices:
		if v1 == v2:
			geodesic[v1,v2] = 0
		else:
			geo_dist = path_length(all_shortest_paths[v1][v2])
			geodesic[v1,v2] = geo_dist

sigma = 10.0

area = dict()
len_bnd = dict()
bnd_con = dict()

for v1 in vertices:
	area[v1] = 0
	len_bnd[v1] = 0
	for v2 in vertices:
		geodesic_dist = geodesic[v1,v2]
		S = math.exp(- pow(geodesic_dist,2)/(2.0 * sigma *sigma))
		area[v1] += S
		print v2
		len_bnd[v1] += S * boundary[v2]
	bnd_con[v1] = len_bnd[v1] / math.sqrt(area[v1])

for v in vertices:
	img_gray[grid == v] = bnd_con[v]

plt.imshow(img_gray)
plt.show()

