# http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb


# Setup environment
from pylab import *
import matplotlib.pyplot as plt
import numpy as np


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
print
# plt.show()


#### Stepping the solver
# show filters before training
if False:
	plt.subplot(2,1,1)
	plt.imshow(solver.net.params['conv1'][0].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5,5*5),
			cmap = 'gray'); axis('off')
	#take 1 step of minibatch SGD
	solver.step(1)
	# check our filters after 1 step- first layer 4x5 grid of 5x5 filters
	plt.subplot(2,1,2)
	plt.imshow(solver.net.params['conv1'][0].diff[:,0].reshape(4,5,5,5).transpose(0,2,1,3).reshape(4*5,5*5),
			cmap = 'gray'); axis('off')
print
# plt.show()


#### Custom Training Loop
# cahnge the solving process by updating the net in a loop
niter = 50 #200
test_interval = 25
#losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

if True:
	print "Start Training Loop"
	# the main solver loop
	for it in range(niter):
		# print "Iteration ", it
		solver.step(1) 	# SGD by caffe

		#store the train loss
		train_loss[it] = solver.net.blobs['loss'].data

		# store the output on the first test batch
		# (start the forward pass at conv1 to avoid loading new data)
		solver.test_nets[0].forward(start = 'conv1')
		output[it] = solver.test_nets[0].blobs['score'].data[:8]

		# run a full test every so often
		# (caffe can also do this for us and write to a log, but we show here
		# how to do it directly in python, where more complicated things are easier)
		if it % test_interval == 0:
			print "Iteration ", it, " testing..."
			correct = 0
			for test_it in range(100):
				solver.test_nets[0].forward()
				correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) ==
								solver.test_nets[0].blobs['label'].data)
			test_acc[it // test_interval] = correct

	#plot the train loss and test accuracy
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(niter), train_loss)
	ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
	plt.show()

	# show the prediction scores evolve over time
	# for i in range(8):
	# 	plt.subplot(2,1,1)
	# 	plt.figure(figsize = (2,2))
	# 	plt.imshow(solver.test_nets[0].blobs['data'].data[i,0], cmap='gray')
	# 	plt.subplot(2,1,2)
	# 	plt.figure(figsize=(10,2))
	# 	plt.imshow(output[:50, i].T, interpolation = 'nearest', cmap='gray')
	# 	plt.xlabel('iteration')
	# 	plt.ylabel('label')
	# plt.show()

print









print
print
print "=========== Caffe_Train_Ex.py DONE ==========="