# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb


# Setup environment
from pylab import *
import matplotlib.pyplot as plt


#### Caffe Setup
# add caffe to python path
import sys
caffe_root = '../../../'
sys.path.insert(0, caffe_root + 'python')
import caffe

#### Setup model
# download training examples for LeNet
import os
os.chdir(caffe_root)
if False:
	os.system('data/mnist/get_mnist.sh')		# download data
	os.system('examples/mnist/create_mnist.sh')	# prepare data
os.chdir('examples')


#### Creating the net 
from caffe import layers as L, params as P

def leNet(lmdb, batch_size): # lmdb is a type of mapped memort database - source file format kinda?
	# a series of linear and simple nonlinear transformations
	n = caffe.NetSpec()

	n.data, n.label = L.Data(batch_size = batch_size, backend = P.Data.LMDB, source = lmdb, 
								transform_param = dict(scale = 1./255), ntop = 2)

	n.conv1 = L.Convolution(n.data, kernel_size = 5, num_output = 20, weight_filler = dict(type = 'xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size = 5, num_output = 50, weight_filler = dict(type = 'xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
	n.fcl = L.InnerProduct(n.pool2, num_output = 500, weight_filler = dict(type = 'xavier'))
	n.relu1 = L.ReLU(n.fcl, in_place = True)
	n.score = L.InnerProduct(n.relu1, num_output = 10, weight_filler = dict(type = 'xavier'))
	n.loss = L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
	f.write(str(leNet('mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
	f.write(str(leNet('mnist/mnist_test_lmdb', 1000)))

os.system('cat mnist/lenet_auto_train.prototxt') # show model structure that we made
print
os.system('cat mnist/lenet_auto_solver.prototxt') # show the learning parameters
print

#### loading and checking the solver
# pick up a SGD solver
caffe.set_mode_cpu()

solver = None
solver = caffe.SGDSolver('mnist/lenet_auto_solver.prototxt')
print

# check feature dimensions - (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print
# just the weight sizes
print [(k, v[0].data.shape) for k, v, in solver.net.params.items()]
print

# run a forward pass and check that they contain our data
solver.net.forward()	# train net
solver.test_nets[0].forward()	# test net (could be more than one)
# tile the first 8 items
plt.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1,0,2).reshape(28, 8*28), 
		cmap = 'gray'); axis('off')
print 'Train labels: ', solver.net.blobs['label'].data[:8]
# plt.show()
plt.imshow(solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1,0,2).reshape(28,8*28),
		cmap = 'gray'); axis('off')
print 'Test labels: ', solver.test_nets[0].blobs['label'].data[:8]
# plt.show()


#### Stepping the solver
# show filters before training
plt.subplot(2,1,1)
plt.imshow(solver.net.params['conv1'][0].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5,5*5),
		cmap = 'gray'); axis('off')
# take 1 step of minibatch SGD
solver.step(1)
# check our filters after 1 step- first layer 4x5 grid of 5x5 filters
plt.subplot(2,1,2)
plt.imshow(solver.net.params['conv1'][0].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5,5*5),
		cmap = 'gray'); axis('off')
plt.show()


#### Custom Training Loop












print
print
print "=========== Caffe_Train_Ex.py DONE ==========="