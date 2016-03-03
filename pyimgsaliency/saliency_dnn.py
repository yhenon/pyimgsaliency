# EXPERIMENTAL CODE


from __future__ import print_function
from scipy.misc import imread, imresize, imsave
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
import h5py
import os
import pdb
import theano
import sys
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD

parser = argparse.ArgumentParser(description='Deep saliency with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')


args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# dimensions of the generated picture.
img_width = 224
img_height = 224

# path to the model weights file.
weights_path = '/home/ubuntu/testing/vgg16_weights.h5'

mean_pixel = [103.939, 116.779, 123.68]

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = cv2.resize(cv2.imread(image_path), (224, 224))
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    img = img.transpose((2,0,1))
    img = np.expand_dims(img, axis=0)

    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    print(x.shape)
    x = x.transpose((1, 2, 0))
    for c in range(3):
        x[:, :, c] = x[:, :, c] + mean_pixel[c]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# this will contain our generated image
raw_saliency= K.placeholder((1, 3, img_width, img_height))

# build the VGG16 network with our raw_saliencyas input
first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
first_layer.input = raw_saliency
model = Sequential()

model.add(first_layer)
#model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu', name='fc_1'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc_2'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax', name='softmax'))

if weights_path:
	model.load_weights(weights_path)

print('Model loaded.')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
#get_feature = theano.function([model.layers[0].input], model.layers[-1].get_output(train=False), allow_input_downcast=True)

x = preprocess_image(base_image_path)

get_feature = theano.function([model.layers[0].input], model.layers[-1].get_output(train=False), allow_input_downcast=True)

feat_orig = np.copy(get_feature(x))
img_orig = np.copy(x)
# get the max response label
max_class = np.argmax(feat_orig)
print("Class label = " + str(max_class))
# build the cost function
gamma = 200.0
weights = np.ones((feat_orig.shape[1]))
weights[max_class] = 0

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# define the loss
loss = K.variable(0.)

output_index = 0
layer_output = model.layers[-1].get_output()

loss += K.sum( layer_output[0,max_class] + (gamma/2) * K.sum(K.square((layer_output[0,:] - feat_orig) * weights)))

# compute the gradients of the raw_saliency wrt the loss
grads = K.gradients(loss, raw_saliency)

outputs = [loss]

if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([raw_saliency], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, 3, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]
    print('loss = ' + str(loss_value))
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    print('gradients = ' + str(grad_values.shape))
    return loss_value, grad_values

for i in range(5):
    (curr_loss,curr_grads) = eval_loss_and_grads(x)
    
    curr_grads = np.maximum(curr_grads,np.zeros(curr_grads.shape))
    
    curr_grads = curr_grads.reshape((1,3,img_width,img_height))

    x -= curr_grads * 30000.0

    fname = result_prefix + '_at_iteration_%d.png' % i

    cv2.imwrite(fname,np.mean(curr_grads[0,:,:,:]*10000000.0,axis=0))

print('x shape = ' + str(x.shape))
print('img_orig shape = ' + str(img_orig.shape))

sal = img_orig[0,:,:,:] - x[0,:,:,:]
sal = np.mean(sal,axis=0)
mean_saliency = np.mean(sal)
#sal[sal < mean_saliency] = 0
sal = np.maximum(sal - mean_saliency,0)
cv2.imwrite('s.png',255*5*sal)
#sal_img = np.dstack((sal,sal,sal))
#sal_img = deprocess_image(sal_img)
#print('sal shape = ' + str(sal_img.shape))
#sal_img = np.mean(sal_img,axis = 2)
#imsave('sal.png',sal_img)
#print(sal_img.shape)
#pdb.set_trace()
