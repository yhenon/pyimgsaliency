import pyimgsaliency
import cv2

filename = '../../netra/testing/bird.jpg'

rbd = pyimgsaliency.get_saliency_rbd(filename).astype('uint8')

ft = pyimgsaliency.get_saliency_ft(filename).astype('uint8')

mbd = pyimgsaliency.get_saliency_mbd(filename).astype('uint8')

img = cv2.imread(filename)

cv2.imshow('img',img)
cv2.imshow('rbd',rbd)
cv2.imshow('ft',ft)
cv2.imshow('mbd',mbd)

cv2.waitKey(0)
