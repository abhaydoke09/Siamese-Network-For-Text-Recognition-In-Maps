"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

def indexToLabel():
    labelDictionary = {}
    cnt = 0
    for i in xrange(26):
      labelDictionary[cnt] = chr(65+i)
      cnt+=1
    for i in xrange(10):
      labelDictionary[cnt] = chr(48+i)
      cnt+=1
    return labelDictionary


# Path to the textfiles for the test set
test_file = 'word_test.txt'
#tf.reset_default_graph()
batch_size = 1
# Place data loading and preprocessing on the cpu
num_classes = 1000
with tf.device('/cpu:0'):
    test_data = ImageDataGenerator(test_file,
                                 mode='inference',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(test_data.data.output_types,
                                       test_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
testing_init_op = iterator.make_initializer(test_data.data)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, [])

# Link variable to model output
score = model.fc8

softmax = tf.nn.softmax(score)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

labels = indexToLabel()
f = open('word_predictions.txt','wb')

# Start Tensorflow session
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('./tmp/finetune_alexnet/checkpoints/model_epoch91.ckpt.meta')
	#saver.restore(sess,'./tmp/finetune_alexnet/checkpoints/model_epoch1.ckpt')
	saver.restore(sess,'./tmp/finetune_alexnet/checkpoints/model_epoch91.ckpt')
	# Validate the model on the entire validation set
	print("{} Starting Testing".format(datetime.now()))
	sess.run(testing_init_op)
	test_acc = 0.
	test_count = 0
	test_data_size = test_data.data_size
	count = 0
	print "Test data size:",test_data_size
	for _ in range(test_data_size):
	    img_batch, label_batch = sess.run(next_batch)
	    probs = sess.run(softmax, feed_dict={x: img_batch,
	                                        y: label_batch,
	                                        keep_prob: 1.})
	    f.write(str(np.argmax(probs)) + ',' + str(np.argmax(label_batch)) + '\n')
	    count += 1
	    if count%1000 == 0:
	    	print count

f.close()
	    
