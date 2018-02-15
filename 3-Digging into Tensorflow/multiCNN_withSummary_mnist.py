#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:55:54 2017
Brunda Chouthoy
CSC 578
Project 3: MultiLayer Convolutional Neural network with Summaries
"""

#import mnist data from tensorflow examples
from tensorflow.examples.tutorials.mnist import input_data
#Load mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Import tensor flow and other required libraries
import tensorflow as tf

sess = tf.InteractiveSession()

#Placeholders
#Create a placeholder for each input image of size 28*28 pixels
x = tf.placeholder(tf.float32, shape=[None, 784])
#Create a placeholder for each of target output where each row is a one-hot 
#10-dimensional vector indicating which digit class the corresponding 
#MNIST image belongs to.
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# tell it where to write info
summaries_dir = '/tmp/mnist_logs'

# Function to create variable summaries. 
# Input arguments: variable, name
# Creates a scope, and then summaries for mean, sd, max, min and histogram.
def variable_summaries(var,name):
  '''Attach a lot of summaries to a Tensor (for TensorBoard visualization)'''
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev' + name, stddev)
    tf.summary.scalar('max' + name, tf.reduce_max(var))
    tf.summary.scalar('min' + name, tf.reduce_min(var))
    tf.summary.histogram('histogram' + name, var)


#Weight initialization
#Function initilizes weights with a small amount of noise
#to prevent 0 gradients
#Input argument: shape 
#Returns the weight variable
def weight_variable(shape):
  #intilize weights with a std deviation of 0.1
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#Function initilizes with a slightly positive initial bias 
#to avoid dead neurons
#Input argument: shape 
#Returns the bias variable
def bias_variable(shape):
  #initialize a positive bias to avoid dead neurons
  initial = tf.constant(0.1, shape=shape) 
  return tf.Variable(initial)

#Convolution and Pooling
#Function applies convolution to layers in the network
#Input arguments: x - input images of size 28*28 pixels and 
#W - weights corresponding to the input matrix'''
def conv2d(x, W):
  #Strides window shifts by 1 in all dimensions and zero padding so that output 
  #is same size as output
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Function applies pooling to layers in the network
#Input argument: x - input images of size 28*28 pixels
def max_pool_2x2(x):
  #max_pool method takes the parameters ksize i.e kernel size with a 2*2 window,
  #strides window shifts and zero padding so that output is same as input
  #max pooling over 2x2 blocks'''
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# Creating the naming scope for the first convolutional layer.
# Adding a name scope will ensure layers are grouped together in the graph logically
with tf.name_scope('Conv1'):  
    #Implementing the First convolutional layer - Convolution computes 32 feautres
    #weight tensor of shape 5*5 patch, 1 input channel and 32 output channels
    W_conv1 = weight_variable([5, 5, 1, 32])
    #Invokes the variable_summaries function to create the variables with summaries
    variable_summaries(W_conv1, 'Conv1/weights')

    #bias vector with a component for each output channel
    b_conv1 = bias_variable([32])
    variable_summaries(b_conv1, 'Conv1/biases')

    #Reshaping the input to 28*28 pixels and 1 color channel
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #For the first convolutional layer - convolve the reshaped input with weight tensor, 
    #apply the relu activation function and add the bias vector
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #Invoke the max pool method to reduce the image size to 14*14
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Conv2'):
    #Second convolutional layer will have 64 features for each 5x5 patch
    #weight tensor with 5*5 patch, 32 input channels and 64 output channels
    W_conv2 = weight_variable([5, 5, 32, 64])
    variable_summaries(W_conv2, 'Conv2/weights')
    
    #bias vector with a component for each output channel
    b_conv2 = bias_variable([64])
    variable_summaries(b_conv2, 'Conv2/biases')
    
    #For the second convolutional layer - convolve the reshaped input with weight tensor,
    #apply the relu activation function and add the bias vector
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #Invoke the max pool method to reduce the image size
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    #Densely connected layer - add a fully-connected layer with 1024 neurons to 
    #allow processing on the entire image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    variable_summaries(W_fc1, 'FulCon1/weights')
    
    #bias vector with a component for each of the 1024 output channels
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1, 'FulCon1/biases')

    #Reshaping the tensor from the pooling layer into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    #Multiply the reshaped tensor with the weight matrix, add the bias component 
    #and apply the relu activation function
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dropout
    #create a placeholder for the probability that a neuron's output is kept during dropout
    keep_prob = tf.placeholder(tf.float32)
    #tf.nn.dropout op handles scaling neuron outputs in addition to masking them, 
    #so dropout should work without any additional scaling
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    #Readout layer
    #Weight matrix for the final layer with shape - 1024*10
    W_fc2 = weight_variable([1024, 10])
    variable_summaries(W_fc2, 'FulCon2/weights')
    
    #bias vector for the readout layer
    b_fc2 = bias_variable([10])
    variable_summaries(b_fc2, 'FulCon2/biases')

    #Multiply the dropout output with weight matrix and add the bias component 
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and evaluate the model
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#Using Adam optimizer for gradient descent to descend entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Calculate the number of correct predictions – using tf.equal to check if our prediction matches the actual true label. 
#Outputs a list of booleans
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#Calculate the percentage of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.summary.merge_all()
#Using FileWriter to add summaries as we train the model
train_writer = tf.summary.FileWriter(summaries_dir + '/train',sess.graph)
#Using FileWriter to add summaries as we test the model
test_writer = tf.summary.FileWriter(summaries_dir + '/test')
tf.global_variables_initializer().run()
 
#Function allows to easily switch between feeding in training or testing instances
def feed_dict(train):
    '''Make a TensorFlow feed_dict: maps data onto Tensor placeholders.'''
    if train:
        xs, ys = mnist.train.next_batch(50, False)
        k = 0.5 # Orig value
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

#Train the model and also write summaries.
#For 20000 training iterations or epochs
for i in range(20000):
    # Every 10th step, measure test-set accuracy, and write test summaries
    if i%10 == 0: 
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        # adding summaries for test
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else: #All other steps, run train_step on training data & add training summaries
        #Code will emit runtime statistics for every 100th step starting at step 99
        if i%100 == 99: 
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # summary is returned by running the session for a train step
            summary, _ = sess.run([merged, train_step],feed_dict=feed_dict(True),
                                  options=run_options,run_metadata=run_metadata)
            # adding metadata about the run
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            # adding training summaries
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else: # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            # add summary of regular step
            train_writer.add_summary(summary, i)

#closing the file writers
train_writer.close()
test_writer.close()

#final test accuracy
print('test accuracy %g' %accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
