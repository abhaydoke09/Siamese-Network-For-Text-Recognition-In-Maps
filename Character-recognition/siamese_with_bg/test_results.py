import numpy as np
from scipy.spatial import distance
import pickle

def get_test_results(words, vectors):
    print("Original word : "+words[0])
    result = []
    for i in range(1, vectors.shape[0]):
        result.append((words[i], distance.euclidean(vectors[0],vectors[i])))

    for tup in sorted(result, key=lambda tup: tup[1]):
        print(tup)
