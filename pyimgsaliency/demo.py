import pyimgsaliency as psal
import cv2

# path to the image
filename = 'bird.jpg'

# get the saliency maps using the 3 implemented methods
rbd = psal.get_saliency_rbd(filename).astype('uint8')

ft = psal.get_saliency_ft(filename).astype('uint8')

mbd = psal.get_saliency_mbd(filename).astype('uint8')

# often, it is desirable to have a binary saliency map
binary_sal = psal.binarise_saliency_map(mbd,method='adaptive')

img = cv2.imread(filename)

cv2.imshow('img',img)
cv2.imshow('rbd',rbd)
cv2.imshow('ft',ft)
cv2.imshow('mbd',mbd)

#openCV cannot display numpy type 0, so convert to uint8 and scale
cv2.imshow('binary',255 * binary_sal.astype('uint8'))


cv2.waitKey(0)
