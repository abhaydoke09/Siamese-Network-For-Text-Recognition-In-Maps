from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
import tensorflow as tf
import numpy as np
import os


from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib.data import Dataset
from helper import get_dataset

import siamese

# prepare data and tf.session
sess = tf.InteractiveSession()

# setup siamese network
siamese_model = siamese.siamese_network();
train_step = tf.train.AdamOptimizer(0.0001).minimize(siamese_model.loss)
saver = tf.train.Saver()
#tf.initialize_all_variables().run()
# if you just want to load a previously trainmodel?

# model_ckpt = 'model.ckpt'
# if os.path.isfile(model_ckpt):
#     input_var = None
#     while input_var not in ['yes', 'no']:
#         input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
#     if input_var == 'yes':
#         new = False


############################ INPUT PIPELINE ############################
train_file = 'word_train_2_classes.txt'
val_file = 'word_val_2_classes.txt'

# train_images, train_labels = get_dataset(train_file)
# val_images, val_labels = get_dataset(val_file)
#
# train_labels_var = tf.constatnt(train_labels)
# train_images_var = tf.constant(train_images)
# val_labels_var = tf.constatnt(val_labels)
# val_images_var = tf.constant(val_images)
#
#
# # create TensorFlow Dataset objects
# tr_data = Dataset.from_tensor_slices((train_images_var, train_labels_var))
# val_data = Dataset.from_tensor_slices((val_images_var, val_labels_var))
#
# # create TensorFlow Iterator object
# iterator = Iterator.from_structure(tr_data.output_types,
#                                    tr_data.output_shapes)
# next_element = iterator.get_next()
#
# # create two initialization ops to switch between the datasets
# training_init_op = iterator.make_initializer(tr_data)
# validation_init_op = iterator.make_initializer(val_data)
###########################################################################


num_classes = 2
batch_size = 10
#Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()
#Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

sess.run(tf.global_variables_initializer())
# start training

#siamese_model.load_initial_weights(sess)
num_epochs = 105
new = False
if new:
    for epoch in range(num_epochs):
        sess.run(training_init_op)
        for step in range(8):
            batch_x1, batch_y1 = sess.run(next_batch)
            batch_x2, batch_y2 = sess.run(next_batch)
            #print(batch_x1, batch_y1)
            #batch_x1, batch_y1 = mnist.train.next_batch(128)
            #batch_x2, batch_y2 = mnist.train.next_batch(128)
            batch_y = (batch_y1 == batch_y2).astype('float')
            #print(batch_y)
            _, loss_v = sess.run([train_step, siamese_model.loss], feed_dict={
                                siamese_model.x1: batch_x1,
                                siamese_model.x2: batch_x2,
                                siamese_model.y_: batch_y})

            if np.isnan(loss_v):
                print('Model diverged with loss = NaN')
                quit()


        print ('epoch %d: loss %.3f' % (epoch, loss_v))

        #   saver.save(sess, 'model.ckpt')
        #     embed = siamese.o1.eval({siamese.x1: mnist.test.images})
        #     embed.tofile('embed.txt')
else:
    saver.restore(sess, 'model.ckpt')
    sess.run(validation_init_op)

    embed = None
    flag = False
    for i in range(4):
        batch_x1, batch_y1 = sess.run(next_batch)
        print(batch_y1)
        if flag:
            print(embed.shape)
            embed = np.vstack((embed,siamese_model.o1.eval({siamese_model.x1: batch_x1})))
        else:
            embed = siamese_model.o1.eval({siamese_model.x1: batch_x1})
            print(embed.shape)
            flag = True
    np.savetxt('embed.txt', embed, delimiter=',')
# # visualize result
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize.visualize(embed, x_test)
