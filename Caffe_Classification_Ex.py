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

# plt.figure(1)
# plt.imshow(image)
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
# for each layer, show the output shape
print "(batch_size, channel_dim, height, width)"
for layer_name, blob in net.blobs.iteritems():
	print layer_name + '\t' +str(blob.data.shape)
print

# output the parameters
print "(output_channels, input_channels, filter_height, filter_width) (output_channels)"
for layer_name, param in net.params.iteritems():
	print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
print


# function to deal with 4D rectangular heatmaps
def vis_square(data):
	# takes an array of shape (n, hiehgt, width) or (n, height, width, 3)
	# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)

	# normalize data for display
	data = (data - data.min()) / (data.max() - data.min())

	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = (((0,n**2 - data.shape[0]),
				(0,1), (0,1))				# add some space between filters
				+((0,0),)*(data.ndim-3))	# dont pad the last dimension (if there is one)
	data = np.pad(data, padding, mode='constant', constant_values=1)	# pad with ones (white)

	# tile the filters into an image
	data = data.reshape((n,n) + data.shape[1:]).transpose((0,2,1,3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n* data.shape[1], n*data.shape[3]) + data.shape[4:])

	plt.imshow(data); plt.axis('off')

# the parameters are a list of [weight, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0,2,3,1))

# applied on the current loaded image (blobs contains the image info) 
feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)
# plt.show()

feat = net.blobs['pool5'].data[0]
vis_square(feat)
# plt.show()

# first plot: Fc layer activations? (output 4096D), secont plot: histogram of activations?....
feat = net.blobs['fc6'].data[0]
# plt.subplot(2,1,1)
# plt.plot(feat.flat)
# plt.subplot(2,1,2)
# _=plt.hist(feat.flat[feat.flat > 0], bins = 100)
# plt.show()

# prediction class probability
feat = net.blobs['prob'].data[0]
# plt.figure(figsize=(15,3))
# plt.plot(feat.flat)
# plt.show()



#### Try your own images
def net_forware_image(image_file):
	# transform it and copy it into the net
	image = caffe.io.load_image(image_file)
	net.blobs['data'].data[...] = transformer.preprocess('data', image)

	# perform classification
	net.forward()

	#obtain probabilities
	output_prob = net.blobs['prob'].data[0]

	#sort the top five predictions from softmax
	top_inds = output_prob.argsort()[::-1][:5]

	# plt.imshow(image)

	print 'probabilities and labels:', zip(output_prob[top_inds], labels[top_inds])

print
print 'Classifying own image'
image_file = '/home/nolanl/Pictures/Panda.jpg'
net_forware_image(image_file)
plt.show()



print
print
print "=========== Caffe_Classification_Ex.py DONE ==========="
