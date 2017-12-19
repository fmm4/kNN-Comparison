
# Refer to Improved nearest neighbor classifiers by weighting and selection of predictors
import numpy as np
import default_knn as knn
import util
from model.Sample import Sample
from operator import itemgetter
import math
from sklearn.metrics.pairwise import rbf_kernel
from decimal import *

class WeightedKnn:
    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    def gaussian(self, v, width):
        # Free parameter
        sigma = 10000
        distance = v/width
        expo = Decimal(math.pow(distance, 2))/ Decimal((2*math.pow(sigma,2)))
        result = Decimal(math.exp(Decimal(-expo)))
        return result

    def distance(self, sample1, sample2):
        s1 = np.array(sample1)
        s2 = np.array(sample2)
        return np.linalg.norm(s1-s2)

    def weight_list(self, samples):
        weightlist = []
        sum_weights = 0
        for k in range(len(samples)):
            distance = samples[k][0]
            gaussian = self.gaussian(distance, len(samples))
            weightlist.append([gaussian, samples[k][1]])
            sum_weights += gaussian

        for k in range(len(weightlist)):
            weightlist[k][0] = weightlist[k][0]/sum_weights
        return weightlist

    def get_distance_list(self,sample):
        distances = []
        for s in self.dataset.samples:
            distances.append([self.distance(sample, s.parameters), s.sampleClass])
        return distances

    def distances_sort(self, distances):
        sorted_distances = sorted(distances, key=itemgetter(0))
        return sorted_distances

    def get_closest_n(self, distances, n):
        sorted_distances = self.distances_sort(distances)
        sliced = sorted_distances[:n]
        return sliced

    def get_votes(self, neighbors):
        classes_dictionary = {}
        class_list = np.unique(np.array([d[1] for d in neighbors]))
        for c in class_list:
            classes_dictionary[c] = 0
        for n in neighbors:
            classes_dictionary[n[1]] += 1
        sorted_dist = util.sort_dict(classes_dictionary, 1)
        return sorted_dist

    def get_weighted_votes(self, neighbors):
        classes_dictionary = {}
        class_list = np.unique(np.array([d[1] for d in neighbors]))
        for c in class_list:
            classes_dictionary[c] = 0
        for n in neighbors:
            classes_dictionary[n[1]] += n[0]
        sorted_dist = util.sort_dict(classes_dictionary, 1)
        return sorted_dist

    def classify(self, sample, n):
        dist_list = self.get_distance_list(sample)
        n_closest_neighbors = self.get_closest_n(dist_list, n)
        weightlist = self.weight_list(n_closest_neighbors)
        most_voted = self.get_weighted_votes(weightlist)
        return most_voted[0]

    def classify_set(self, set, knn):
        result = 0
        for k in set.samples:
            predicted_class = self.classify(k.parameters,knn)
            if predicted_class[0] == k.sampleClass:
                result += 1
        result = float(result) / float(len(set.samples))
        return result