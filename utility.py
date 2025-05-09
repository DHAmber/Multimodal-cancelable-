import numpy as np
from config import *
def hamming_distance(bitstring1, bitstring2):
    return sum(b1 != b2 for b1, b2 in zip(bitstring1, bitstring2))


def dice_coefficient(a, b):
    intersection = np.sum(a * b)
    return 2 * intersection / (np.sum(a) + np.sum(b))

def getlevals(lebels):
    result_lebel=[]
    new_labels = np.array([np.eye(number_of_user)[i] for i in range(number_of_user)])
    for l in lebels:
        index=int(l.split('_')[1])
        result_lebel.append(new_labels[index-1])
    return np.array(result_lebel)

def group_features_by_user(features, labels):
    grouped_features = []
    grouped_labels = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        user_indices = [i for i, l in enumerate(labels) if l == label]
        user_features = features[user_indices]
        averaged_feature = np.mean(user_features, axis=0)
        thrash_mean=np.mean(averaged_feature)
        averaged_feature =(averaged_feature >= thrash_mean).astype(int)
        grouped_features.append(averaged_feature)
        grouped_labels.append(label)

    return np.array(grouped_features), grouped_labels