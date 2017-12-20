import os

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./tmp/finetune_alexnet/checkpoints/model_epoch101.ckpt.meta')
        #saver.restore(sess,'./tmp/finetune_alexnet/checkpoints/model_epoch1.ckpt')
        saver.restore(sess,'./tmp/finetune_alexnet/checkpoints/model_epoch101.ckpt')
        # Validate the model on the entire validation set
        print("{} Starting Testing".format(datetime.now()))
        
        weights_dict = {}
	layer_names = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            if var.name.split('/')[0] not in layer_names:
                layer_names.append(var.name.split('/')[0])
	print(layer_names)
   	for layer in layer_names:    	
            weights_dict[layer] = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,layer))
        
	np.save("trained_weights_with_binary_bg.npy", weights_dict)
