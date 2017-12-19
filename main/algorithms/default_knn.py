# Default implementation of brute force KNN from Scipy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import *
from util import split_train_test
import util


class default_knn:
    samples = None
    knn = None
    scores = None
    n = -1
    k_max = -1
    k_min = -1

    def __init__(self, dataset, min_k, max_k):
        self.samples = dataset
        self.k_min = min_k
        self.k_max = max_k
        self.n = self.find_global_accuracies()
        self.knn = KNeighborsClassifier(n_neighbors=self.n, algorithm='brute')
        self.knn.fit(self.samples.get_X(), self.samples.get_y())

    def find_global_accuracies(self):
        scores = {}
        ten_folds = self.samples.n_folds(10)
        if self.k_max - self.k_min == 0:
            return 1
        for i in range(self.k_min, self.k_max):
            global_accuracy = 0
            for fold in range(0, len(ten_folds)):
                test, train = util.split_train_test(ten_folds, fold)
                temp_knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute')
                temp_knn.fit(train.get_X(), train.get_y())
                score = temp_knn.score(test.get_X(), test.get_y())
                global_accuracy += float(score)
            global_accuracy /= 10
            scores[i] = global_accuracy
        result = util.sort_dict(scores, 1)

        return result[0][0]

    def get_score(self, dataset):
        return self.knn.score(dataset.get_X(), dataset.get_y()), self.n
