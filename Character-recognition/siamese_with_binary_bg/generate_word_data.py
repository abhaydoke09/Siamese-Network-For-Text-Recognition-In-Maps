import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import cv2
#import imutils
import random
import re
from tqdm import tqdm
import matplotlib.image as mpimg
import itertools
from siamese_bg_util import getImageFromWord

background_images = []
for i in range(1, 2):
    img = mpimg.imread('./maps/map_crop_0' + str(i) + '.jpg')
    background_images.append(img)


def get_random_crop():
    image_number = random.randint(1, 1)
    img = background_images[image_number - 1]
    # print(img.shape)
    height, width = img.shape[0], img.shape[1]
    start_row = random.randint(0, height - 227)
    start_column = random.randint(0, width - 227)
    new_img = img[start_row:start_row + 227, start_column:start_column + 227, :]
    new_img = new_img[..., list(list(itertools.permutations([0, 1, 2]))[random.randint(0, 5)])]
    # print new_img.shape
    return new_img


def merge_background_text(img, bg_image):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0 and img[i, j, 1] == 0 and img[i, j, 2] == 0:
                bg_image[i, j, :] = 0
    return bg_image


def getWordList():
    word_list = []
    f = open('high_frequency_english_words.txt', 'rb')
    word_list = f.readlines()
    f.close()

    for i, line in enumerate(word_list):
        word_list[i] = line.lower()
        word_list[i] = re.sub(r'[^a-zA-Z0-9]', '', line)
        word_list[i] = re.sub(r'[\n\r]', '', line)
        word_list[i] = line.lower()

    return word_list


def changeCase(word):
    case = random.randint(0, 2)

    # Explicitly getting capital words. Uncomment following line to get all types.
    #case = 2

    if case == 0:
        # get lowercase word
        return word.lower()
    elif case == 1:
        # get camelcase word
        return word[0].upper() + word[1:]
    else:
        # get uppercase word
        return word.upper()


def generateData():
    word_list = getWordList()
    # print word_list
    trainFile = open('word_train_with_bg.txt', 'wb')
    valFile = open('word_val_with_bg.txt', 'wb')
    testFile = open('word_test_bg.txt', 'wb')

    SAVE_DIR = './word_image_data_bg/'
    classLabel = 0
    # word_list = ["the"]
    for word in tqdm(word_list):
        word = re.sub(r'[^a-zA-Z0-9]', '', word)
        for i in xrange(150):
            img = getImageFromWord(word)
            cv2.imwrite(SAVE_DIR + word + str(i) + '.png', img)
            if i < 100:
                trainFile.write('./word_image_data_bg/' + word + str(i) + '.png' + ' ' + str(classLabel) + '\n')
            elif i >= 100 and i < 120:
                valFile.write('./word_image_data_bg/' + word + str(i) + '.png' + ' ' + str(classLabel) + '\n')
            else:
                testFile.write('./word_image_data_bg/' + word + str(i) + '.png' + ' ' + str(classLabel) + '\n')
        classLabel += 1
    trainFile.close()
    valFile.close()
    testFile.close()

generateData()

