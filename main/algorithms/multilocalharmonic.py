# Filipe Mendes Mariz
# A new k-harmonic nearest neighbor classifier based on the multi-local means (Zhibin Pan, Yidi Wang, Weiping Ku)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import *
from util import split_train_test
from operator import itemgetter
from util import sort_dict, split_train_test


class mlmkhnn:
    trainset = None
    class_sets = None
    neighbor = -1
    k_min = -1
    k_max = -1

    def __init__(self, dataset, min_k, max_k):
        self.trainset = dataset
        self.class_sets = self.init_class_sets(self.trainset)
        self.k_min, self.k_max = min_k, max_k
        self.neighbor = self.find_best_k()


    def init_class_sets(self, trainset):
        new_class_sets = {}
        samples = trainset.samples
        classes = np.unique(np.array([d.sampleClass for d in samples]))
        for w in classes:
            new_class_sets[w] = []
        for sample in samples:
            new_class_sets[sample.sampleClass].append(sample.parameters)
        return new_class_sets

    def get_precision(self, testset):
        if self.class_sets == None:
            self.class_sets = self.init_class_sets(self.trainset)
        precision = 0.0
        for sample in testset.samples:
            predicted = self.predict(sample.parameters, self.neighbor)
            if predicted == sample.sampleClass:
                precision += 1.0
        return precision / len(testset.samples), self.neighbor

    def get_harmonic_mean_distance(self, sample, class_samples):
        k = len(class_samples)
        divisor = 0.0
        for local_mean in class_samples:
            distance = self.distance(sample, local_mean)
            if distance != 0.0:
                divisor += 1.0 / distance
        return float(k) / float(divisor)

    def get_harmonic_divisor_list(self, sample, class_samples):
        harmonic_list = []
        for local_mean in class_samples:
            distancia = self.distance(sample, local_mean)
            if distancia != 0:
                harmonic_list += [(1.0 / distancia)]
        return harmonic_list

    def find_local_mean(self, samples):
        mean = np.array(samples[0][1])
        for i in range(1, len(samples)):
            mean += np.array(samples[i][1])
        mean = mean / len(samples)
        return mean.tolist()

    def find_nearest_of_class(self, w, parameters, k):
        distance_set = self.get_distance_list(parameters, w)
        return self.get_closest_n(distance_set, k)

    def get_distance_list(self, parameters, subset):
        distances = []
        for sample in subset:
            distancia = self.distance(parameters, sample)
            if distancia != 0.0:
                distances.append([(self.distance(parameters, sample)), sample])
        return distances

    def distance(self, sample1, sample2):
        s1 = np.array(sample1)
        s2 = np.array(sample2)
        return np.linalg.norm(s1 - s2)

    def get_closest_n(self, distances, n):
        sorted_distances = sorted(distances, key=itemgetter(0))
        sliced = sorted_distances[:n]
        return sliced

    def find_best_k(self):
        folds = self.trainset.n_folds(10)
        all_results = {}
        for o in range(0, 10):
            test, train = split_train_test(folds, o)
            all_results = self.addDictionaries(all_results,
                                               self.calculate_best_precision(train, test,self.k_max))
        all_results = sort_dict(all_results, 1)
        best_result = all_results[0][1]
        all_results.reverse()
        for k in all_results:
            if k[1] == best_result:
                return k[0]
        return -1

    def addDictionaries(self, dict1, dict2):
        if dict1 == {}:
            return dict2
        else:
            for w in dict2:
                if w not in dict1:
                    dict1[w] = 0.0
                dict1[w] += dict2[w]
        return dict1

    def calculate_best_precision(self, train, test, k):
        class_sets = self.init_class_sets(train)
        results = {}
        for s in test.samples:
            parametros = s.parameters
            classe = s.sampleClass
            closest_class_local_vector = {}
            for w in class_sets:
                distance_list = {}
                distance_list[w] = self.find_nearest_of_class(class_sets[w], parametros, k)
                list_of_means = self.find_all_means(distance_list[w])
                closest_class_local_vector[w] = list_of_means
            class_harmonic_distance_list = {}
            for w in closest_class_local_vector:
                if len(closest_class_local_vector[w]) != 0:
                    class_harmonic_distance_list[w] = self.get_harmonic_divisor_list(parametros,
                                                                                 closest_class_local_vector[w])
            for i in range(1, k):
                class_harmonic_distances = self.get_harmonic_distance_precision(class_harmonic_distance_list, i)
                sorted_dictionary = sort_dict(class_harmonic_distances, 1)
                sorted_dictionary.reverse()
                if classe == sorted_dictionary[0][0]:
                    if i not in results:
                        results[i] = 0.0
                    results[i] += 1.0
        for i in results:
            results[i] = results[i] / len(test.samples)
        return results

    def get_harmonic_distance_precision(self, class_divisors, i):
        distance_list = {}
        for w in class_divisors:
            sum_class_divisors = sum(class_divisors[w][:i])
            if sum_class_divisors != 0.0:
                distance_list[w] = 1.0 / sum(class_divisors[w][:i])
        return distance_list

    def find_all_means(self, subset):
        mean_vectors_list = []
        for i in range(0, len(subset)):
            mean_vectors_list += [self.find_local_mean(subset[:i + 1])]
        return mean_vectors_list

    def predict(self, sample, k):
        class_harmonic_distances = {}
        subsets = {}
        for w in self.class_sets:
            nn_set = self.find_nearest_of_class(self.class_sets[w], sample, k)
            local_mean_vectors = self.find_all_means(nn_set)
            subsets[w] = local_mean_vectors
        for w in subsets:
            class_harmonic_distances[w] = self.get_harmonic_mean_distance(sample, subsets[w])
        sorted_dictionary = sort_dict(class_harmonic_distances, 1)
        sorted_dictionary.reverse()
        return sorted_dictionary[0][0]
