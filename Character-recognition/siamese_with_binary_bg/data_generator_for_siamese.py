"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import random
import re

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import cv2
#from util import getImageFromWord
from siamese_bg_util import getImageFromWord

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.batch_size = batch_size

        # retrieve the data from the text file
        self._read_txt_file()

        self._read_words()


        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
             self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        self.data = data
        
        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_threads=8,
                      output_buffer_size=100*batch_size)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_threads=8,
                      output_buffer_size=100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data


    def _read_words(self):
        f = open("US_Cities.txt")
        lines = f.readlines()
        f.close()
        self.words = []
        for line in lines:
            self.words.append(re.sub(r'[^a-zA-Z0-9]','',line))
        self.num_words = len(self.words)

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        self.image_dict = {}
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))
                if int(items[1]) in self.image_dict:
                    self.image_dict[int(items[1])].append(items[0])
                else:
                    self.image_dict[int(items[1])] = [items[0]]
        #print(self.image_dict)

    def get_run_time_batch(self):
        batch_x1_data = []
        batch_x2_data = []
        batch_y1 = []
        batch_y2 = []

        for i in range(self.batch_size):
            word1 = self.words[random.randint(0,self.num_words-1)]
            word2 = word1
            x1_label = 0
            x2_label = 0
            flag = random.randint(0,10)
            if flag>3:
                x2_label = 1
                word2 = self.words[random.randint(0,self.num_words-1)]
            batch_x1_data.append(getImageFromWord(word1))
            batch_x2_data.append(getImageFromWord(word2))
            batch_y1.append(x1_label)
            batch_y2.append(x2_label)

        return np.array(batch_x1_data), np.array(batch_y1), np.array(batch_x2_data), np.array(batch_y2)


    def get_test_batch(self, size):
        test_word = self.words[random.randint(0, self.num_words - 1)]
        batch_x1_data = []
        batch_y1 = []
        test_words = []
        color_images = []
        img, color = getImageFromWord(test_word)
        batch_x1_data.append(img)
        color_images.append(color)
        test_words.append(test_word)
        batch_y1.append(0)
        for i in range(size-1):
            prob = random.randint(0,9)
            if prob>7:
                img, color = getImageFromWord(test_word)
                batch_x1_data.append(img)
                color_images.append(color)
                batch_y1.append(0)
                test_words.append(test_word)
            else:
                word2 = self.words[random.randint(0,self.num_words-1)]
                img, color = getImageFromWord(word2)
                batch_x1_data.append(img)
                color_images.append(color)
                batch_y1.append(1)
                test_words.append(word2)
        return np.array(batch_x1_data), np.array(color_images),np.array(batch_y1), test_words



    def get_batch_for_siamese_network(self):
        number_of_labels = len(self.image_dict.keys())
        batch_x1 = []
        batch_x2 = []
        batch_y1 = []
        batch_y2 = []
        for i in range(self.batch_size):
            x1_label = random.randint(0, number_of_labels-1)
            x2_label = x1_label
            flag = random.randint(0,10)
            if flag>3:
                x2_label = random.choice([label for label in range(number_of_labels) if label != x1_label])
            batch_y1.append(x1_label)
            batch_y2.append(x2_label)
            batch_x1.append(self.image_dict[x1_label][random.randint(0, len(self.image_dict[x1_label]) - 1)])
            batch_x2.append(self.image_dict[x2_label][random.randint(0, len(self.image_dict[x2_label]) - 1)])
        # same_image_count = 0
        # for i in range(self.batch_size):
        #     #print(batch_y1[i], batch_y2[i])
        #     print(batch_x1[i], batch_x2[i])
        #     if batch_y1[i] == batch_y2[i]:
        #         same_image_count+=1
        # print(same_image_count)
        batch_x1_data = []
        batch_x2_data = []
        for i in range(self.batch_size):
            img1 = cv2.imread(batch_x1[i])
            img1 = cv2.resize(img1, (227, 227))
            #print(img1.shape)
            img1 = img1[:,:,::-1]
            batch_x1_data.append(img1)
            img2 = cv2.imread(batch_x2[i])
            img2 = cv2.resize(img2, (227, 227))
            img2 = img2[:, :, ::-1]
            batch_x2_data.append(img2)
        return np.array(batch_x1_data), np.array(batch_y1), np.array(batch_x2_data), np.array(batch_y2)



    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        #one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        #one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, label
