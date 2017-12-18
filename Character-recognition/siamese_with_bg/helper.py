import re
import tensorflow as tf
def get_dataset(filename):
    f = open(filename)
    lines = f.readlines()
    f.close()

    images = []
    labels = []

    for line in lines:
        line = re.sub(r'[^a-zA-Z0-9///. ]', '', line)
        img, label = line.split(" ")
        label = int(label)
        images.append(img)
        labels.append(label)

    return (images, labels)


def input_parser(img_path, label):
    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)
    return img_decoded, label