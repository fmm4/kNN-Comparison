# Refer to A Proposal for Local k Values for k-Nearest Neighbor Rule
import numpy as np
import default_knn as knn
import util
from testing.timer import Timer
from model.Sample import Sample
from operator import itemgetter
import math
from sklearn.neighbors import KNeighborsClassifier


class local_best_k:
    samples = []
    dataset = []
    scores = []
    vizinhanca = []
    k_min = -1
    k_max = -1

    def __init__(self, dataset, min_k, max_k):
        self.dataset = dataset
        self.k_min = min_k
        self.k_max = max_k
        self.find_global_accuracies()
        self.init_vizinhanca()

        for k in dataset.samples:
            self.samples += [[k, -1]]

    def insert(self, array, sampleclass):
        np.append(self.samples, [Sample(array, sampleclass), -1], axis=0)

    def add_all(self, array):
        for k in array:
            self.insert(k[0], k[1])

    def distance(self, sample1, sample2):
        s1 = np.array(sample1)
        s2 = np.array(sample2)
        return np.linalg.norm(s1 - s2)

    def init_vizinhanca(self):
        vizinhancas = []
        for i in range(0, len(self.dataset.samples)):
            amostra = self.dataset.samples[i]
            vizinhanca = self.get_lista_distancias(amostra.parameters, i)
            vizinhancas.append(vizinhanca)
        self.vizinhanca = vizinhancas

    def get_lista_distancias(self, parametros, skip):
        vizinhanca = []
        for i in range(0, len(self.dataset.samples)):
            if i != skip:
                distance = self.distance(parametros, self.dataset.samples[i].parameters)
                s_class = self.dataset.samples[i].sampleClass
                vizinhanca.append([distance, s_class, i])
        vizinhanca = sorted(vizinhanca, key=itemgetter(0))
        return vizinhanca

    def classify(self, sample, k_voters_n, k_min, k_max):
        vizinhos = self.get_lista_distancias(sample, -1)
        k_voters = vizinhos[:k_voters_n]
        optimal_local_k = self.find_optimal_k(k_voters, k_min, k_max)
        chosen_neighbors = vizinhos[:optimal_local_k]
        classes = set(s_c[1] for s_c in chosen_neighbors)
        count = {}
        for k in classes:
            count[k] = 0
        for k in chosen_neighbors:
            count[k[1]] += 1
        return max(count, key=count.get), optimal_local_k

    def find_optimal_k(self, sample_set, k_min, k_max):
        k = 0.0
        for i in sample_set:
            if self.samples[i[2]][1] != -1:
                k += self.samples[i[2]][1]
            else:
                optimalk = self.find_local_k(i[2], k_min)
                self.samples[i[2]][1] = optimalk
                k += optimalk
        return int(k / len(sample_set))

    def classify_set(self, dataset):
        right = 0
        full_set = len(dataset.samples)
        opt_k_set = []
        for k in dataset.samples:
            guessed_class, opt_k = self.classify(k.parameters, 3, self.k_min, self.k_max)
            opt_k_set.append(opt_k)
            true_class = k.sampleClass
            if guessed_class == true_class:
                right += 1.0
        return float(right) / float(full_set), opt_k_set

    # Finds optimal k-neighbor for specific sample, refer to algorithm 1 on paper
    def find_local_k(self, sample, k_min):
        acc_set = {}
        distan_list = self.vizinhanca[sample]
        for k in range(k_min, len(self.scores)):
            sliced = distan_list[:k]
            count = 0.0
            for distances in sliced:
                if distances[1] == self.samples[sample][0].sampleClass:
                    count += 1.0
            acc_set[k] = ((float(count) / float(k)) + self.scores[k]) / 2
        sorted_dist = util.sort_dict(acc_set, 1)
        return sorted_dist[0][0]

    def find_global_accuracies(self):
        scores = {}
        ten_folds = self.dataset.n_folds(10)
        k_min = self.k_min
        k_max = self.k_max
        for i in range(k_min, k_max):
            global_accuracy = 0
            for fold in range(0, len(ten_folds)):
                test, train = util.split_train_test(ten_folds, fold)
                temp_knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute')
                temp_knn.fit(train.get_X(), train.get_y())
                score = temp_knn.score(test.get_X(), test.get_y())
                global_accuracy += float(score)
            global_accuracy /= 10
            scores[i] = global_accuracy
        self.scores = scores
