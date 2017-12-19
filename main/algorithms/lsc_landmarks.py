import numpy as np
from sklearn.cluster import KMeans

# Refer to Efficient kNN classification algorithm for big data
# and Large Scale Spectral Clustering with Landmark-Based Representation
import numpy as np
from numpy import dot
import default_knn as knn
import util
from util import dbg as d
from model.Sample import Sample
from operator import itemgetter
import random
import scipy.sparse.linalg as sp

class LSCLandmarks:
    dataset = None

    def __init__(self, dataset):
        self.dataset = dataset

    def get_LSC_random(self, cluster_number, r):
        data_points = self.dataset.samples
        landmarks = self.get_landmarks_random(cluster_number)
        sparse_matrix = self.get_sparse_affinity_matrix(data_points, landmarks, r)
        row_sum = self.get_row_sum_vector(sparse_matrix)  # Zn = D^(-1/2)Z

        # This is Zn, it should be samples x landmark size
        final_z = row_sum * np.transpose(sparse_matrix)


        # Calculate the Singular Value Decomposition of the final_z
        V_t, E, U = np.linalg.svd(final_z, False)

        U_t = np.transpose(U)
        I = np.eye(cluster_number)
        E_minus1 = np.power(E, -1)
        E_minus1 = E_minus1*I

        final_a = np.dot(E_minus1, np.dot(U, np.transpose(final_z)))
        result = np.transpose(final_a)
        d("Size " + str(len(final_a)) + " - " + str(len(final_a[0])))




    # Good results, not viable on very large datasets
    def get_landmarks_kmean(self,kmean,number):
        new_kmean = None
        samples = self.dataset.get_X()
        for i in range(0, kmean):
            new_kmean = KMeans(int(number)).fit(samples)
            samples = np.array(new_kmean.cluster_centers_)
        return new_kmean

    # Viable on large datasets, maybe not the best results
    def get_landmarks_random(self, number):
        number_samples = len(self.dataset.samples)
        returned_items = range(number_samples)
        random.shuffle(returned_items)
        returned_items = returned_items[:number]
        returned_samples = []
        for k in returned_items:
            returned_samples.append(self.dataset.samples[k])
        return returned_samples

    def gaussian_kernel(self, array1, array2, bandwith):
        x = np.array(array1.parameters)
        z = np.array(array2.parameters)
        dividend = np.subtract(x, z)  # Xi - Zj
        dividend = np.linalg.norm(dividend)  # || Xi - Zj ||
        dividend = np.multiply(dividend, dividend)  # || Xi - Zj ||^2
        dividend = np.multiply(-1, dividend)  # -(|| Xi - Zj || ^2)
        divisor = bandwith * bandwith  # (h ^2)
        divisor = divisor * 2 # 2 * (h ^ 2 )
        return np.exp(dividend/divisor)  # exp(-(|| Xi - Zj || ^2)/(2*h^2))


    # Nadaraya-Watson kernel regression
    def get_sparse_affinity_matrix(self, data_points, landmarks, r_nearest_neighbor):
        w = np.empty([r_nearest_neighbor,len(data_points)])
        for i in range(len(data_points)):
            submatrix_z = self.get_r_closest(data_points[i], landmarks, r_nearest_neighbor)
            kernels = np.empty(len(submatrix_z))
            for j in range(len(submatrix_z)):
                kernels[j] = self.gaussian_kernel(data_points[i], submatrix_z[j], 1)
            k_sum = np.sum(kernels)
            for j in range(len(kernels)):
                w[j][i] = kernels[j]/k_sum
        return w

    def get_r_closest(self, k, landmarks, r):
        xi = np.array(k.parameters)
        distances = {}
        for i in range(len(landmarks)):
            distances[i] = np.linalg.norm(np.subtract(xi, np.array(landmarks[i].parameters)))
        array = util.sort_dict(distances, 1)
        array.reverse()
        array = array[:r]
        return [landmarks[i] for i in [k[0] for k in array]]

    def get_row_sum_vector(self, matrix):
        number_of_rows = len(matrix)
        sum_vector = np.empty(number_of_rows)
        for k in range(number_of_rows):
            sum_vector[k] = sum(matrix[k])
        return np.power(sum_vector, -1/2)

    def compute_v(self, diag, u, weights):
        diag_i = np.power(diag, -1)  # sum ^ -1
        u_t = np.transpose(u)
        v_t = diag_i *  np.matmul(u_t,weights)
        return np.transpose(v_t)