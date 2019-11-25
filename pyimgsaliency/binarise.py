import numpy as np

def binarise_saliency_map(saliency_map,method='adaptive',threshold=0.5):

	# check if input is a numpy array
	if type(saliency_map).__module__ != np.__name__:
		print('Expected numpy array')
		return None

	#check if input is 2D
	if len(saliency_map.shape) != 2:
		print('Saliency map must be 2D')
		return None

	if method == 'fixed':
		return (saliency_map > threshold)

	elif method == 'adaptive':
		adaptive_threshold = 2.0 * saliency_map.mean()
		return (saliency_map > adaptive_threshold)

	elif method == 'clustering':
		print('Not yet implemented')
		return None

	else:
		print("Method must be one of fixed, adaptive or clustering")
		return None
