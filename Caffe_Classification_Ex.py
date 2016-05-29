# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

#### General Setup 
import numpy as np
import matplotlib.pyplot as plt
import time

# display plots in this notebook
# %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10,10) 		# large images
plt.rcParams['image.interpolation'] = 'nearest'	# dont interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'				# use grayscale output rather than a (potentially misleading) colour heatmap


#### Caffe Setup
# add caffe to python path
import sys
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')

import caffe


#### Get trained model Setup
# downloads CaffeNet if u dont have it
import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
	print 'CaffeNet found (already downloaded).'
else:
	print "Downloading pre-trained CaffeNet model..."
	os.system(caffe_root + '/scripts/download_model_binary.py ' + caffe_root + 'models/bvlc_reference_caffenet')


#### Setup model
caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,		#defines the structure of the mdoel
				model_weights,	# contains the trained weights
				caffe.TEST)		# use test mode (e.g. dont perform dropout) - leave out nodes during testinf to help prevent overfitting
print

#### Input Preprocessing
# load the mean imagenet image (part of Caffe distribution) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1) # average over pixels to obtain the mean pixel value (across 2D)
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))		# move image channels to outermost dimension
transformer.set_mean('data', mu)				# subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)			# rescale from [0,1] to [0,255]
transformer.set_channel_swap('data', (2,1,0))	# swap channels from RGB to BGR
print

#### CPU classification
# set the size of the input
net.blobs['data'].reshape(	50,		# batch size
							3,		# 3-channel (BGR) images
							227,227)# image size of 227x227

# load an image
imageFile = caffe_root + 'examples/images/cat.jpg'
image = caffe.io.load_image(imageFile)
transformed_image = transformer.preprocess('data', image)

plt.figure(1)
plt.imshow(image)
# plt.figure(2)
# plt.imshow(transformed_image)
# plt.show()

# classify the image
net.blobs['data'].data[...] = transformed_image # copy image to memory allocated for the net

# the actual classification
output = net.forward()
output_prob = output['prob'][0] # the output probability vector for the first image in the batch

# print 'Probability vector: ', np.sort(output_prob)
prob_class = output_prob.argmax()
print 'Predicted class is: ', prob_class

# check against the imagetnet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
	os.system(caffe_root + 'data/ilsvrc12/get_ilsvrc_aux.sh')	

labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label: ', labels[prob_class] # tabby cat should be correct

# show top 5 selections
top_inds = output_prob.argsort()[::-1][:5] # reverse sort and take 5 largest items
print 'Probabilities and labels: ', zip(output_prob[top_inds], labels[top_inds])
print 


#### Examining Intermediate outputs in the network





print
print
print "=========== Caffe_Classification_Ex.py DONE ==========="
