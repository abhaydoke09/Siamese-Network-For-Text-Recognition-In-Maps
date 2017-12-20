import numpy as np
import random
import cv2
from scipy.spatial import distance
import pickle

def get_test_results(words, vectors):
    print("Original word : "+words[0])
    result = []
    for i in range(1, vectors.shape[0]):
        result.append((words[i], distance.euclidean(vectors[0],vectors[i]), i))
    print(sorted(result, key=lambda tup: tup[1]))
    return sorted(result, key=lambda tup: tup[1])


def changeCase(word):
    case = random.randint(0, 2)

    # Explicitly getting capital words. Uncomment following line to get all types.
    case = 2

    if case == 0:
        # get lowercase word
        return word.lower()
    elif case == 1:
        # get camelcase word
        return word[0].upper() + word[1:]
    else:
        # get uppercase word
        return word.upper()

def getImageFromWord(word):
    word = changeCase(word)

    height, width = 512, 512
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :, :] = 255

    font = random.choice([0, 1, 2, 3, 4]) | 16 if random.randint(0, 1) else 0
    bottomLeftCornerOfText = (30, 150)
    fontScale = 5
    fontColor = (0, 0, 0)
    lineType = random.randint(2, 4)
    thickness = 4
    lineType = 8

    while True:
        textsize = cv2.getTextSize(word, font, fontScale, lineType)[0]
        if textsize[0] < width - 20:
            break
        else:
            fontScale -= 1

    # print textsize

    # get coords based on boundary
    textX = (img.shape[1] - textsize[0]) / 2
    textY = (img.shape[0] + textsize[1]) / 2

    # add text centered on image
    cv2.putText(img, word, (textX, textY), font, fontScale, fontColor, lineType)

    rotateFlag = random.randint(0, 1)
    if rotateFlag:
        rotateAngle = random.randint(-10, 10)
        M = cv2.getRotationMatrix2D((width / 2, height / 2), rotateAngle, 1)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    affineFlag = random.randint(0, 1)
    if affineFlag:
        pts1 = np.float32([[10, 10], [200, 50], [50, 200]])
        pts2 = np.float32([[10 + random.randint(-20, 20), 30 + random.randint(-20, 20)]
                              , [200, 50],
                           [50 + random.randint(-20, 20), 200 + random.randint(-20, 20)]])

        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))

    img = cv2.resize(img, (227, 227))
    ##bg_image = get_random_crop()
    ##bg_image = merge_background_text(img, bg_image)
    # print(bg_image.shape)
    # img = np.add(img,bg_image)
    # plt.imshow(img)
    return img