import skfuzzy as fuzzy
import numpy as np
from model.Dataset import Dataset
from model.Sample import  Sample
from operator import itemgetter
from sklearn import svm

class svm_cknn:
    initial_dataset = None
    dataset = None
    centers = None
    clustered_dataset = None

    def __init__(self, dataset):
        self.dataset = dataset
        self.initial_dataset = dataset

    def cluster_centers_pcm(self, numero_de_centros, vizinhos):
        parameters = np.transpose(np.array([k.parameters for k in self.initial_dataset.samples]))
        cntr, u, u0, d, jm, p, fpc = fuzzy.cluster.cmeans(
            parameters,
            numero_de_centros,
            2.0,
            0.005,
            2000)
        new_samples = []
        local_dataset = self.dataset
        for k in cntr:
            amostras = self.get_k1_closest(k, vizinhos)
            for sample_pega in amostras:
                new_samples.append(sample_pega)
        self.clustered_dataset = Dataset("Clustered", new_samples)

    def get_k1_closest(self, center, k1):
        dist_list = []
        samples = self.dataset.samples
        for k in range(0, len(samples)):
            distance = self.distance(center, samples[k].parameters)
            dist_list.append([distance, k])
        dist_list = sorted(dist_list, key=itemgetter(0))
        chosen_neighbors = dist_list[:k1]
        returned_neighbor = []
        #  Deleta amostras que forem retornadas para nao haver duplicados
        #  Artigo nunca especifica se e ou nao para haver esse pruning
        for k in chosen_neighbors:
            chosen_sample = self.dataset.samples[k[1]]

            returned_neighbor.append(chosen_sample)
        return returned_neighbor

    def classify(self, sample, k_n):
        dist_list = []
        samples = self.dataset.samples
        for k in range(0, len(samples)):
            distance = self.distance(sample, samples[k].parameters)
            s_class = samples[k].sampleClass
            dist_list.append([distance, s_class])
        dist_list = sorted(dist_list, key=itemgetter(0))
        chosen_neighbors = dist_list[:k_n]
        classes = set(s_c[1] for s_c in chosen_neighbors)
        count = {}
        for k in classes:
            count[k] = 0
        for k in chosen_neighbors:
            count[k[1]] += 1
        return max(count, key=count.get)

    def distance(self, s1, s2):
        return np.linalg.norm(s1-s2)

# def function_J(self,samples, centers):
#     N = len(samples)
#     M = len(centers)
#     sumv = 0
#     for i in range(N):
#         for j in range(M):
#             xi = samples[i]
#             cj = centers[j]
#             dij = self.function_d(samples[i],centers[j])
#             uij = function_u(xi,cj)
#
# def function_u(self,x,c,centers):
#     divisor = self.function_u_divisor(x,c,centers)
#     return
#
# def function_u_divisor(self,x,c,centers):
#     upper = self.function_d(x,c)
#     sumv = 0
#     for k in range(len(centers)):
#         lower = self.function_d(x,centers[k])
#         sumv += upper/lower
#     sumv = np.power(sumv, 2/(self.m-1))
#     return sumv
#
# def function_c(self,):
#
#
# def function_d(self,X,C):
#     p1 = np.array(X)
#     p2 = np.array(C)
#     return np.linalg.norm(p1-p2)