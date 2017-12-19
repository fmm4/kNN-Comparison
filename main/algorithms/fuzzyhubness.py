# Filipe Mendes Mariz
# A new k-harmonic nearest neighbor classifier based on the multi-local means (Zhibin Pan, Yidi Wang, Weiping Ku)
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from model import *
from util import split_train_test
from operator import itemgetter
from util import sort_dict
from math import sqrt, exp


class fuzzyhubness:
    amostras = None
    classes = None
    hubness = None
    globals = None
    local1 = None
    local2 = None
    laplace = 0.001
    nrClasses = 0
    freqClasseVizinho = None
    freqAparicao = None
    freqClasseParaClasse = None
    bThetha = -1
    bK = -1
    bType = ""

    def __init__(self, dataset, min_k, max_k):
        self.classes = np.unique([k.sampleClass for k in dataset.samples])
        self.nrClasses = len(self.classes)
        self.inicializar_dicionario(dataset)
        self.inicializar_distancias()
        self.inicializarFreqAparicao(min_k, max_k + 1)
        self.inicializarFreqClasseVizinho(min_k, max_k + 1)
        self.inicializar_hubness(min_k, max_k + 1)
        self.inicializar_global(min_k, max_k + 1)
        self.inicializar_local1(min_k, max_k + 1)
        self.inicializar_local2(min_k, max_k + 1)
        self.inicialize_variables(10, min_k, max_k + 1)

    def inicialize_variables(self, limiarMax, min_k, max_k):
        best = {}
        best["acc"] = -1.0
        best["type"] = ""
        best["k"] = -1
        best["thetha"] = -1
        for thethaLimiar in range(0, limiarMax):
            for k in range(min_k, max_k):
                acuracias = {}
                acuracias["crisp"] = 0.0
                acuracias["local1"] = 0.0
                acuracias["local2"] = 0.0
                acuracias["global"] = 0.0
                for amostra in self.amostras:
                    classe = self.amostras[amostra]["classe"]
                    parametros = self.amostras[amostra]["parametros"]
                    classe_predita_crisp = self.predict(parametros, k, thethaLimiar, "crisp")
                    if classe == classe_predita_crisp:
                        acuracias["crisp"] += 1.0
                    classe_predita_global = self.predict(parametros, k, thethaLimiar, "global")
                    if classe == classe_predita_global:
                        acuracias["global"] += 1.0
                    classe_predita_local1 = self.predict(parametros, k, thethaLimiar, "local1")
                    if classe == classe_predita_local1:
                        acuracias["local1"] += 1.0
                    classe_predita_local2 = self.predict(parametros, k, thethaLimiar, "local2")
                    if classe == classe_predita_local2:
                        acuracias["local2"] += 1.0
                resultados_ordenados = sort_dict(acuracias, 1)
                if resultados_ordenados[0][1] >= best["acc"]:
                    best["acc"] = resultados_ordenados[0][1]
                    best["k"] = k
                    best["type"] = resultados_ordenados[0][0]
                    best["thetha"] = thethaLimiar
        self.bThetha = best["thetha"]
        self.bK = best["k"]
        self.bType = best["type"]

    # def inicializarFreqClasseVizinho(self, min_k, max_k):
    #     tamanho = len(self.amostras[0]["vizinhos"])
    #     self.freqClasseVizinho = {}
    #     for amostra in self.amostras:
    #         self.freqClasseVizinho[amostra] = {}
    #         for k in range(min_k, max_k):
    #             classes_vizinhos = self.get_class_set(amostra)
    #             votos_amostra = self.contar_strings_classe(k, classes_vizinhos)
    #             self.freqClasseVizinho[amostra][k] = votos_amostra
    #     print self.freqClasseVizinho

    def inicializarFreqClasseVizinho(self, min_k, max_k):
        self.freqClasseVizinho = {}
        for amostra in self.amostras:
            self.freqClasseVizinho[amostra] = {}
            for k in range(min_k, max_k):
                self.freqClasseVizinho[amostra][k] = {}
                for classe in self.classes:
                    self.freqClasseVizinho[amostra][k][classe] = 0.0
        for amostra in self.amostras:
            vizinhos = self.amostras[amostra]["vizinhos"]
            for k in range(min_k, max_k):
                for vizinho in vizinhos[:k]:
                    self.freqClasseVizinho[vizinho[0]][k][self.amostras[amostra]["classe"]] += 1.0

    def inicializarFreqAparicao(self, min_k, max_k):
        self.freqAparicao = {}
        for amostra in self.amostras:
            self.freqAparicao[amostra] = {}
            for k in range(min_k, max_k):
                self.freqAparicao[amostra][k] = 0.0
        for amostra in self.amostras:
            vizinhos = self.amostras[amostra]["vizinhos"]
            for k in range(min_k, max_k):
                for vizinho in vizinhos[:k]:
                    self.freqAparicao[vizinho[0]][k] += 1.0

    def inicializar_hubness(self, min_k, max_k):
        totallaplace = self.nrClasses * self.laplace
        hub = {}
        self.hubness = {}
        for amostra in self.amostras:
            self.hubness[amostra] = {}
            for k in range(min_k, max_k):
                self.hubness[amostra][k] = {}
                nk = self.freqAparicao[amostra][k]
                for classe in self.classes:
                    nkc = self.freqClasseVizinho[amostra][k][classe]
                    self.hubness[amostra][k][classe] = (nkc + self.laplace) / (nk + totallaplace)

    def get_hubnesses(self, nao_contado, mean, std_dev):
        conta = self.contar_strings_classe(len(nao_contado), nao_contado)
        hubnesses = {}
        for classe in self.classes:
            hubnesses[classe] = 0.0
            BNk = conta[classe]
            if (std_dev[classe] != 0.0):
                hubnesses[classe] = (BNk - mean[classe]) / std_dev[classe]
            else:
                hubnesses[classe] = 0.0
        return hubnesses

    def get_mean(self, vizinhos, k):
        contar = {}
        for classe in self.classes:
            contar[classe] = 0
        index = 0
        for amostra in vizinhos:
            if index < k:
                count = self.contar_strings_classe(k, vizinhos[amostra])
                for classe in count:
                    contar[classe] += count[classe]
            index += 1
        for classe in contar:
            contar[classe] /= k
        return contar

    def contar_strings_classe(self, k, vizinhos):
        count = {}
        for classe in self.classes:
            count[classe] = 0.0
        index = 0.0
        contado = vizinhos[:k]
        for classe in contado:
            count[classe] += 1.0
        return count

    def get_std(self, lista_vizinhos, media, k):
        std = {}
        for classe in self.classes:
            std[classe] = 0.0
        powered = {}
        for classe in self.classes:
            powered[classe] = 0.0
            index = 0.0
            for amostra in lista_vizinhos:
                if index < k:
                    contagem = self.contar_strings_classe(k, lista_vizinhos[amostra])
                    powered[classe] += pow(contagem[classe] - media[classe], 2)
                index += 1.0
        for v in powered:
            powered[v] = sqrt(powered[v] / k)
        return powered

    def get_class_set(self, amostra):
        class_count = []
        for v in [self.amostras[c[0]]["classe"] for c in self.amostras[amostra]["vizinhos"]]:
            class_count.append(v)
        return class_count

    def inicializar_local1(self, min_k, max_k):
        for k in range(min_k, max_k):
            for amostra in self.amostras:
                if "local1" not in self.amostras[amostra]:
                    self.amostras[amostra]["local1"] = {}
                vizinhos = self.amostras[amostra]["vizinhos"][:k + 1]
                votos = {}
                for classe in self.classes:
                    votos[classe] = 0.0
                for v in vizinhos:
                    votos[self.amostras[v[0]]["classe"]] += 1.0
                self.amostras[amostra]["local1"][k] = {}
                for voto in votos:
                    self.amostras[amostra]["local1"][k][voto] = (self.laplace + votos[voto]) / (
                        len(self.classes) * self.laplace + k + 1.0)

    def inicializar_local2(self, min_k, max_k):
        for k in range(min_k, max_k):
            for amostra in self.amostras:
                if "local2" not in self.amostras[amostra]:
                    self.amostras[amostra]["local2"] = {}
                self.amostras[amostra]["local2"][k] = {}
                for classe in self.amostras[amostra]["local1"][k]:
                    if self.amostras[amostra]["classe"] == classe:
                        self.amostras[amostra]["local2"][k][classe] = 0.51 + 0.49 * self.amostras[amostra]["local1"][k][
                            classe]
                    else:
                        self.amostras[amostra]["local2"][k][classe] = 0.49 * self.amostras[amostra]["local1"][k][classe]

    def predict_set(self, test_set):
        type = self.bType
        thetha = self.bThetha
        k = self.bK
        precisao = 0.0
        for amostra in test_set.samples:
            classe = amostra.sampleClass
            parametros = amostra.parameters
            predicted = self.predict(parametros, k, thetha, type)
            if predicted == classe:
                precisao += 1.0
        return precisao / len(test_set.samples), self.bK, self.bThetha

    def predict(self, parametros, vizinhos, limiar, tipo):
        vizinholist = self.get_lista_distancia(parametros)[:vizinhos]
        votos = {}
        k = len(vizinholist)
        for classe in self.classes:
            votos[classe] = 0
        for v in vizinholist:
            votos_amostra = self.hubness[v[0]][k]
            classe_amostra = self.amostras[v[0]]["classe"]
            if self.freqAparicao[v[0]][k] > limiar:
                for classe in votos:
                    votos[classe] += votos_amostra[classe]
            else:
                for classe in votos:
                    if tipo == "crisp":
                        if classe == classe_amostra:
                            votos[classe] += (self.laplace + 1.0) / (1.0 + len(self.classes) * self.laplace)
                        else:
                            votos[classe] += self.laplace / (1.0 + (len(self.classes) * self.laplace))
                    if tipo == "global":
                        votos[classe] += self.classToClassPriorsAllK[k][classe][classe_amostra]
                    if tipo == "local1":
                        votos[classe] += self.amostras[v[0]]["local1"][k][classe]
                    if tipo == "local2":
                        votos[classe] += self.amostras[v[0]]["local2"][k][classe]
        result = sort_dict(votos, 1)
        return result[0][0]

    def inicializar_global(self, kMin, kMax):
        self.globals = {}
        classDataKNeighborRelationAllK = {}
        classToClassPriorsAllK = {}
        for k in range(kMin, kMax):
            classDataKNeighborRelationAllK[k] = {}
            classToClassPriorsAllK[k] = {}
            for classe in self.classes:
                classDataKNeighborRelationAllK[k][classe] = {}
                for amostra in self.amostras:
                    classDataKNeighborRelationAllK[k][classe][amostra] = 0
            for classe in self.classes:
                classToClassPriorsAllK[k][classe] = {}
                for classe2 in self.classes:
                    classToClassPriorsAllK[k][classe][classe2] = 0
        classHubnessSumAllK = {}
        for k in range(kMin, kMax):
            classHubnessSumAllK[k] = {}
            for classe in self.classes:
                classHubnessSumAllK[k][classe] = 0.0
            for amostra in self.amostras:
                classe = self.amostras[amostra]["classe"]
                classDataKNeighborRelationAllK[k][classe][amostra] = 0
                for i in range(0, k):
                    vizinho = self.amostras[amostra]["vizinhos"][i][0]
                    if vizinho not in classDataKNeighborRelationAllK[k][classe]:
                        classDataKNeighborRelationAllK[k][classe][vizinho] = 0.0
                    classDataKNeighborRelationAllK[k][classe][vizinho] += 1.0
                    classToClassPriorsAllK[k][self.amostras[vizinho]["classe"]][classe] += 1.0
                    classHubnessSumAllK[k][self.amostras[vizinho]["classe"]] += 1.0
            laplacetotal = self.laplace * len(self.classes)
            for i in self.classes:
                for j in self.amostras:
                    classDataKNeighborRelationAllK[k][i][j] += self.laplace
                    classDataKNeighborRelationAllK[k][i][j] /= (
                        self.freqAparicao[j][k] + 1 + laplacetotal)
            for i in self.classes:
                for j in self.classes:
                    classToClassPriorsAllK[k][i][j] += self.laplace
                    classToClassPriorsAllK[k][i][j] /= (
                        classHubnessSumAllK[k][j] + laplacetotal)
        self.classToClassPriorsAllK = classToClassPriorsAllK

    def inicializar_dicionario(self, dataset):
        amostras = {}
        for sample in range(len(dataset.samples)):
            amostra = {}
            amostra["hubness"] = {}
            amostra["classe"] = dataset.samples[sample].sampleClass
            amostra["parametros"] = dataset.samples[sample].parameters
            amostras[sample] = amostra
        self.amostras = amostras

    def inicializar_distancias(self):
        for k in self.amostras:
            lista = self.get_lista_distancia(self.amostras[k]["parametros"])
            self.amostras[k]["vizinhos"] = lista

    def get_lista_distancia(self, parametros):
        lista_distancias = []
        for k in self.amostras:
            distancia = self.distance(parametros, self.amostras[k]["parametros"])
            if distancia != 0:
                lista_distancias.append([k, distancia])
        lista_distancias = sorted(lista_distancias, key=itemgetter(1))
        return lista_distancias

    def distance(self, sample1, sample2):
        s1 = np.array(sample1)
        s2 = np.array(sample2)
        return np.linalg.norm(s1 - s2)
