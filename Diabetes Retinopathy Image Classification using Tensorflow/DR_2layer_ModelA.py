#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
CSC 578 - Neural Networks and Deep Learning 
Final Project Fall 2017
Brunda Chouthoy
Diabetic Retinopathy Image classification with multi-layer CNNs using Tensorflow
Model A
"""
#Importing required libraries
import numpy as np
import cv2
import os
import pandas as pd
import tensorflow as tf

#Setting dimension values for the image
imgWidth = 128 
imgHeight = 128 
nChannels = 3

#No of output classes
outputDim = 5
batchSize = 100
#Total number of epochs to run
numEpochs = 10

#Parse data and Set directory input paths 
trainDir = '/Users/Bru/Desktop/578Final/data/train'
trainLabels = '/Users/Bru/Desktop/578Final/data/trainLabels.csv'
valDir = '/Users/Bru/Desktop/578Final/data/val'
valLabels = '/Users/Bru/Desktop/578Final/data/validationLabels.csv'


'''
Function reads the train/validation images with labels
Inputs: train or validation directory path and flag indication train or validation
Outputs: numpy array with image names and image labels/classes
'''
def getImageNames(path, flag):
    if flag:
        #train set
        trainCsv = pd.read_csv(path)
        headers = trainCsv.columns
        #Returns an array with train images and labels 
        return np.array([trainCsv[headers[0]], trainCsv[headers[1]]])
    else:
        #validation set
        valCsv = pd.read_csv(path)
        headers = valCsv.columns
        #Return an array with validation image names and labels
        return np.array([valCsv[headers[0]], valCsv[headers[1]]])    
    
'''
Function reads and transforms images to the specified image dimensions
Input: Image path
Output: numpy array of the transformed image
'''       
def transformImages(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (imgWidth, imgHeight))
    #Return an array with the transformed image
    return np.array(img).reshape((imgHeight, imgWidth, nChannels))


'''
Function creates a batch for the specified batch size and batch number
Inputs: batch number, input train/val data, batch size and path of the train/val directory
Outputs: an array of images and labels as a batch
'''
def createCustomBatch(batch_num, array, batch_size, path):
    #if batch_num == 0:
        #Randomly shuffle the array before creating batches
        #shuffle(array)
    #Dividing the data into sets/batches 
    batch = array[0][(batch_num * batch_size): ((batch_num+1)*batch_size)]
    data_batch = []
    #Iterate over images in each batch
    for j, image_name in enumerate(batch):
        try:
            if path == trainDir: #train data path
                imagePath = '{}.jpeg'.format(os.path.join(path, image_name))
                data_batch.append((transformImages(imagePath), array[1][j+(batch_num * batch_size)]))
            elif path == valDir: #val data path
                imagePath = '{}.jpeg'.format(os.path.join(path, image_name))
                data_batch.append((transformImages(imagePath), array[1][j+(batch_num * batch_size)]))
        except:
            print('Error reading: {}'.format(imagePath))
    return np.array(data_batch)

'''
Function applies convolution to the layers in the network
Inputs: input images and weights corresponding to the input matrix
'''      
def conv2d(x, W):
    #Strides window shifts by 1 in all dimensions and zero padding so that output is same size as output
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''Function applies pooling to layers in the network
Input: input images 
'''
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_2layer(x):
    weights  = {
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 128], stddev=0.1)),
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=0.1)),
        'W_fc1': tf.Variable(tf.random_normal([32*32*128, 128], stddev=0.1)),
        'W_out': tf.Variable(tf.random_normal([128, outputDim], stddev=0.1))
        }
    
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([128])),
        'b_conv2': tf.Variable(tf.random_normal([128])),
        'b_fc1': tf.Variable(tf.random_normal([128])),
        'b_out': tf.Variable(tf.random_normal([outputDim]))
    }
    
    keep_rate = 0.8
    
    # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 128, 128, 3])
    # Convolution Layer 1
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (Down sampling)
    conv1 = maxpool2d(conv1)
    
    # Convolution Layer 2
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer
    flatten = tf.reshape(conv2, [-1, 32*32*128])
    fc1 = tf.nn.relu(tf.matmul(flatten, weights['W_fc1']) + biases['b_fc1'])
    fc1 = tf.nn.dropout(fc1, keep_rate)
    
    output = tf.nn.relu(tf.matmul(fc1, weights['W_out']) + biases['b_out'])
    return output

    
'''
Function converts image labels to one hot encoded arrays
Input: train/validation labels
Output: one hot encoded array of labels
'''
def transformLabels(imgLabels):
    max = np.max(imgLabels) 
    oneHotEncoded = np.zeros([imgLabels.shape[0], max+1])
    for i, y in enumerate(imgLabels):
        #one hot encoding for 5 classes
        oneHotEncoded[i, y] = 1
    return oneHotEncoded

'''
Function transforms and creates two separate arrays for images and labels
Inputs: a batch array with train/val images and labels
Output: Returns transformed images and labels for the batch input
'''
def getBatchXY(batchArray):
    #Reshaping the images with the specified input dimension of the image
    x = np.array([x for x in batchArray[:, 0]]).reshape([-1, imgWidth, imgHeight, nChannels])
    #Invoking transformLabels function to get one hot encoded labels
    y = transformLabels(batchArray[:, 1])
    return x, y

'''
Function trains the Neural network with Tensorflow session
'''
def train_neural_network():
    
    x = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, nChannels])
    y = tf.placeholder(tf.float32, [None,outputDim])
    keep_prob = tf.placeholder(tf.float32)
    
    prediction = cnn_2layer(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    trainImages = getImageNames(trainLabels,1)
    valImages = getImageNames(valLabels,0)
    print('Number of training images: {}\nNumber of Validation images: {}'.format(len(trainImages[0]),(len(valImages[0]))))

    numBatches = int(len(trainImages[0])/batchSize)
    
    #Generating validation images and labels
    valBatch = createCustomBatch(0, valImages,len(valImages[0]), valDir)
    batch_vx,batch_vy = getBatchXY(valBatch)
    
    #Start a tensorflow session
    sess = tf.Session()
    #Initializing all global variables
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(numEpochs):
        avg_cost = 0
        accrArray = [] 
        for i in range(numBatches): 
            trainBatch = createCustomBatch(i, trainImages, batchSize, trainDir)
            batch_x, batch_y = getBatchXY(trainBatch)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            avg_cost += sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})/numBatches
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            accrArray.append(train_accuracy)

        print ("Epoch: %03d/%03d cost: %.9f" % (epoch+1, numEpochs, avg_cost))
        print ('Training accuracy: %.5f' % (np.average(accrArray)))
        val_accuracy = sess.run(accuracy, feed_dict={x: batch_vx, y: batch_vy,keep_prob: 1.0})
        print ("Validation accuracy:  %.5f" % (val_accuracy))
    
    sess.close()
    print('Ending train NN')
        
train_neural_network()
