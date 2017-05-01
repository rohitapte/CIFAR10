import numpy as np
import pickle
from sklearn import preprocessing
import random
import math
import os
from six.moves import urllib
import tarfile
import tensorflow as tf

NUM_FILE_BATCHES=5

#LEARNING_RATE = 0.0001
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY=0.1
NUM_GENS_TO_WAIT=250.0

TRAINING_ITERATIONS = 30000
DROPOUT = 0.5
BATCH_SIZE = 500
VALIDATION_SIZE = 0
IMAGE_TO_DISPLAY = 10

#first download the data
data_dir='data'
if not os.path.exists(data_dir):
	os.makedirs(data_dir)
cifar10_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

data_file=os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
	pass
else:
	def progress(block_num, block_size, total_size):
		progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
		print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
	filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
	tarfile.open(filepath, 'r:gz').extractall(data_dir)

#with open('../data/CIFAR10/batches.meta',mode='rb') as file:
with open('data/cifar-10-batches-py/batches.meta',mode='rb') as file:
	batch=pickle.load(file,encoding='latin1')
label_names=batch['label_names']


def load_cifar10data(filename):
	with open(filename,mode='rb') as file:
		batch=pickle.load(file,encoding='latin1')
		features=batch['data'].reshape((len(batch['data']),3,32,32)).transpose(0,2,3,1)
		labels=batch['labels']
		return features,labels

x_train=np.zeros(shape=(0,32,32,3))
train_labels=[]
for i in range(1,NUM_FILE_BATCHES+1):
	#ft,lb=load_cifar10data('../data/CIFAR10/data_batch_'+str(i))
	ft,lb=load_cifar10data('data/cifar-10-batches-py/data_batch_'+str(i))
	x_train=np.vstack((x_train,ft))
	train_labels.extend(lb)


unique_labels=list(set(train_labels))
lb=preprocessing.LabelBinarizer()
lb.fit(unique_labels)
y_train=lb.transform(train_labels)

#x_test,test_labels=load_cifar10data('../data/CIFAR10/test_batch')
x_test_data,test_labels=load_cifar10data('data/cifar-10-batches-py/test_batch')
y_test=lb.transform(test_labels)

def updateImage(x_train_data,distort=True):
	x_temp=x_train_data.copy()
	x_output=np.zeros(shape=(0,32,32,3))
	for i in range(0,x_temp.shape[0]):
		temp=x_temp[i]
		if distort:
			if random.random()>0.5:
				temp=np.fliplr(temp)
			brightness=random.randint(-63,63)
			temp=temp+brightness
			contrast=random.uniform(0.2,1.8)
			temp=temp*contrast
		mean=np.mean(temp)
		stddev=np.std(temp)
		temp=(temp-mean)/stddev
		temp=np.expand_dims(temp,axis=0)
		x_output=np.append(x_output,temp,axis=0)
	return x_output

x_test=updateImage(x_test_data,False)

def truncated_normal_var(name, shape, dtype):
	return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
def zero_var(name, shape, dtype):
	return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

x=tf.placeholder(tf.float32,shape=[None,x_train.shape[1],x_train.shape[2],x_train.shape[3]],name='x')
labels=tf.placeholder(tf.float32,shape=[None,y_train.shape[1]],name='labels')
keep_prob=tf.placeholder(tf.float32,name='keep_prob')

with tf.variable_scope('conv1') as scope:
	conv1_kernel=truncated_normal_var(name='conv1_kernel',shape=[5,5,3,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv1=tf.nn.conv2d(x,conv1_kernel,strides,padding='SAME')
	conv1_bias=zero_var(name='conv1_bias',shape=[64],dtype=tf.float32)
	conv1_add_bias=tf.nn.bias_add(conv1,conv1_bias)
	relu_conv1=tf.nn.relu(conv1_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool1=tf.nn.max_pool(relu_conv1,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer1')
norm1=tf.nn.lrn(pool1,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm1')
norm1 = tf.nn.dropout(norm1, keep_prob)

with tf.variable_scope('conv2') as scope:
	conv2_kernel=truncated_normal_var(name='conv2_kernel',shape=[5,5,64,64],dtype=tf.float32)
	strides=[1,1,1,1]
	conv2=tf.nn.conv2d(norm1,conv2_kernel,strides,padding='SAME')
	conv2_bias=zero_var(name='conv2_bias',shape=[64],dtype=tf.float32)
	conv2_add_bias=tf.nn.bias_add(conv2,conv2_bias)
	relu_conv2=tf.nn.relu(conv2_add_bias)

pool_size=[1,3,3,1]
strides=[1,2,2,1]
pool2=tf.nn.max_pool(relu_conv2,ksize=pool_size,strides=strides,padding='SAME',name='pool_layer2')
norm2=tf.nn.lrn(pool2,depth_radius=5,bias=2.0,alpha=1e-3,beta=0.75,name='norm2')
norm2 = tf.nn.dropout(norm2, keep_prob)

reshaped_output=tf.reshape(norm2, [-1, 8*8*64])
reshaped_dim=reshaped_output.get_shape()[1].value

#with tf.variable_scope('full1') as scope:
full_weight1=truncated_normal_var(name='full_mult1',shape=[reshaped_dim,1024],dtype=tf.float32)
full_bias1=zero_var(name='full_bias1',shape=[1024],dtype=tf.float32)
full_layer1=tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight1),full_bias1))
full_layer1=tf.nn.dropout(full_layer1,keep_prob)

#with tf.variable_scope('full2') as scope:
full_weight2=truncated_normal_var(name='full_mult2',shape=[1024, 256],dtype=tf.float32)
full_bias2=zero_var(name='full_bias2',shape=[256],dtype=tf.float32)
full_layer2=tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight2),full_bias2))
full_layer2=tf.nn.dropout(full_layer2,keep_prob)

#with tf.variable_scope('full3') as scope:
full_weight3=truncated_normal_var(name='full_mult3',shape=[256,IMAGE_TO_DISPLAY],dtype=tf.float32)
full_bias3=zero_var(name='full_bias3',shape=[IMAGE_TO_DISPLAY],dtype=tf.float32)
final_output=tf.add(tf.matmul(full_layer2,full_weight3),full_bias3,name='final_output')

logits=tf.identity(final_output,name='logits')

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels),name='cross_entropy')
#train_step=tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
generation_run = tf.Variable(0, trainable=False,name='generation_run')
model_learning_rate=tf.train.exponential_decay(LEARNING_RATE,generation_run,NUM_GENS_TO_WAIT,LEARNING_RATE_DECAY,staircase=True,name='model_learning_rate')
train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(final_output,1),tf.argmax(labels,1),name='correct_prediction')
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='accuracy')


epochs_completed=0
index_in_epoch = 0
num_examples=x_train.shape[0]

init=tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	def next_batch(batch_size):
		global x_train
		global y_train
		global index_in_epoch
		global epochs_completed
		start = index_in_epoch
		index_in_epoch += batch_size

		if index_in_epoch > num_examples:
			# finished epoch
			epochs_completed += 1
			# shuffle the data
			perm = np.arange(num_examples)
			np.random.shuffle(perm)
			x_train=x_train[perm]
			y_train=y_train[perm]
			# start next epoch
			start = 0
			index_in_epoch = batch_size
			assert batch_size <= num_examples
		end = index_in_epoch
		#return x_train[start:end], y_train[start:end]
		x_output=updateImage(x_train[start:end],True)
		
		return x_output,y_train[start:end]



	# visualisation variables
	train_accuracies = []
	validation_accuracies = []
	x_range = []

	display_step=1

	for i in range(TRAINING_ITERATIONS):

		#get new batch
		batch_xs, batch_ys = next_batch(BATCH_SIZE)

		# check progress on every 1st,2nd,...,10th,20th,...,100th... step
		if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
			train_accuracy = accuracy.eval(feed_dict={x:batch_xs,labels: batch_ys,keep_prob: 1.0})
			validation_accuracy=0.0
			for j in range(0,x_test.shape[0]//BATCH_SIZE):
				validation_accuracy+=accuracy.eval(feed_dict={ x: x_test[j*BATCH_SIZE : (j+1)*BATCH_SIZE],labels: y_test[j*BATCH_SIZE : (j+1)*BATCH_SIZE],keep_prob: 1.0})
			validation_accuracy/=(j+1.0)
			print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
			validation_accuracies.append(validation_accuracy)
			#print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
			train_accuracies.append(train_accuracy)
			x_range.append(i)
			# increase display_step
			if i%(display_step*10) == 0 and i:
				display_step *= 10
		# train on batch
		sess.run(train_step, feed_dict={x: batch_xs, labels: batch_ys, keep_prob: DROPOUT})

	#validation_accuracy = accuracy.eval(feed_dict={x: x_test,y_: y_test,keep_prob: 1.0})
	#print('validation_accuracy => %.4f'%validation_accuracy)
	validation_accuracy=0.0
	for j in range(0,x_test.shape[0]//BATCH_SIZE):
		validation_accuracy+=accuracy.eval(feed_dict={ x: x_test[j*BATCH_SIZE : (j+1)*BATCH_SIZE],labels: y_test[j*BATCH_SIZE : (j+1)*BATCH_SIZE],keep_prob: 1.0})
	validation_accuracy/=(j+1.0)
	print('validation_accuracy => %.4f'%validation_accuracy)
	#print(sess.run(tf.all_variables()))
	saver=tf.train.Saver()
	save_path=saver.save(sess,'./CIFAR10_model')
	sess.close()