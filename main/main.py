from algorithms.default_knn import default_knn
from algorithms.local_best import local_best_k
from algorithms.svm_cknn import *
from algorithms.util import *
from algorithms.testing.timer import Timer
from algorithms.weighted_knn import WeightedKnn
from algorithms.multilocalharmonic import mlmkhnn
from algorithms.fuzzyhubness import fuzzyhubness
from scipy.stats import f_oneway
from random import shuffle
from copy import deepcopy
import os
import errno

clock = Timer()
lista = ["GLOBAL_K", "LOCAL_BEST_K", "WEIGHTED_HARMONIC", "FUZZY_HUBNESS"]
modelos = ["GLOBAL_K", "LOCAL_BEST_K", "WEIGHTED_HARMONIC", "FUZZY_HUBNESS"]


def main():
    # iris_db = read_database("iris", 4)
    # wine_db = read_database("wine", 0)
    # credit_db = read_xls_database("creditcard")
    # landsat_db = read_landsat()
    yeast_db = read_yeast()
    # iono_db = read_ionosphere()
    # glass_db = read_glass()

    apply_test(yeast_db, 30, 15, 3, 1, 20)


def imprimir_resultados(database, resultados):
    imprimido = "Results - " + database.getNome() + "\n"
    info = database.get_info()
    imprimido += "DB Info:"
    imprimido += "\nSamples :" + str(info["samples"])
    imprimido += "\nClasses :" + str(info["classes"])
    imprimido += "\nParameters :" + str(info["parameters"])
    imprimido += "\n----------------------------------------\n"
    ordered = sort_dict(resultados, 0)
    for modelo in modelos:
        if modelo != "FUZZY_HUBNESS":
            imprimido += "[ " + modelo + " ] - P: " + precisions_to_string(
                resultados[modelo]["precisao"]) + ", K: " + k_to_string(modelo, resultados[modelo][
                "melhor_k"]) + " (Train: " + times_to_string(
                resultados[modelo]["tempo_treino"]) + ", Test: " + times_to_string(
                resultados[modelo]["tempo_teste"]) + ")\n"
        else:
            imprimido += "[ " + modelo + " ] - P: " + precisions_to_string(
                resultados[modelo]["precisao"]) + ", K: " + k_to_string(modelo, resultados[modelo][
                "melhor_k"]) + " T: " + thetha_to_string(
                resultados[modelo]["melhor_thetha"]) + " (Train: " + times_to_string(
                resultados[modelo]["tempo_treino"]) + ", Test: " + times_to_string(
                resultados[modelo]["tempo_teste"]) + ")\n"
        if modelo != "GLOBAL_K":
            imprimido += str(
                f_oneway(resultados["GLOBAL_K"]["precisao"], resultados[modelo]["precisao"]))+"\n"
    imprimido += "----------------------------------------\n"

    return imprimido


def apply_test(dataset, n_of_folds, cluster_centers=3, cluster_neighbors=6, k_min=1, k_max=10):
    banco = dataset.samples
    shuffle(banco)
    used_db = Dataset(dataset.getNome(), banco)
    n_of_centers = cluster_centers
    n_of_neighbors = cluster_neighbors
    n_of_folds = n_of_folds
    fold_set = used_db.n_folds(n_of_folds)
    training_time = []
    datasets_clustered = []
    datasets_default = []

    for fold_escolhido in range(n_of_folds):
        test, train = split_train_test(fold_set, fold_escolhido)
        pcm = svm_cknn(train)
        clock.tick()
        pcm.cluster_centers_pcm(n_of_centers, n_of_neighbors)
        training_time.append(float(clock.tock()))
        c_train = pcm.clustered_dataset
        datasets_clustered.append([c_train, test])
        datasets_default.append([train, test])

    fatores = {"precisao": [],
               "tempo_treino": [],
               "tempo_teste": [],
               "melhor_k": [],
               "melhor_thetha": []}

    resultados = {}
    for modelo in modelos:
        resultados[modelo] = deepcopy(fatores)

    resultados_clustered = deepcopy(resultados)

    k_minimo = k_min
    k_maximo = k_max

    print "Progress " + str(float(0) * 100 / float(n_of_folds)) + "%..."
    for fold_escolhido in range(n_of_folds):
        # ###Default###
        train = datasets_default[fold_escolhido][0]
        test = datasets_default[fold_escolhido][1]

        for modelo in modelos:
            precisao, tempo_treino, tempo_teste, k_escolhido, thetha_escolhido = apply_model(train, test, modelo,
                                                                                             k_minimo,
                                                                                             k_maximo)
            resultados[modelo]["precisao"].append(precisao)
            resultados[modelo]["tempo_treino"].append(tempo_treino)
            resultados[modelo]["tempo_teste"].append(tempo_teste)
            resultados[modelo]["melhor_k"].append(k_escolhido)
            if thetha_escolhido is not None:
                resultados[modelo]["melhor_thetha"].append(thetha_escolhido)

        ###Clustered###
        train = datasets_clustered[fold_escolhido][0]

        for modelo in modelos:
            precisao, tempo_treino, tempo_teste, k_escolhido, thetha_escolhido = apply_model(train, test, modelo,
                                                                                             k_minimo,
                                                                                             k_maximo)
            resultados_clustered[modelo]["precisao"].append(precisao)
            resultados_clustered[modelo]["tempo_treino"].append(tempo_treino)
            resultados_clustered[modelo]["tempo_teste"].append(tempo_teste)
            resultados_clustered[modelo]["melhor_k"].append(k_escolhido)
            if thetha_escolhido is not None:
                resultados_clustered[modelo]["melhor_thetha"].append(thetha_escolhido)

        print "Progress " + str(float(fold_escolhido + 1) * 100 / float(n_of_folds)) + "%..."

    resultados_normais = imprimir_resultados(dataset, resultados)
    tempo_clustering = ("Time to Apply Clustering: " + times_to_string(training_time))
    resultados_clustered = imprimir_resultados(dataset, resultados_clustered)

    nome_arquivo = "resultados_novos/resultados_" + dataset.getNome() + ".txt"
    if not os.path.exists(os.path.dirname(nome_arquivo)):
        try:
            os.makedirs(os.path.dirname(nome_arquivo))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    impressao = resultados_normais + "\n" + tempo_clustering + "\n" + resultados_clustered
    with open(nome_arquivo, "w") as f:
        f.write(str(impressao))


def apply_model(train, test, tipo, min_k, max_k):
    traintime = 0
    score = 0
    k = 0
    testtime = 0
    thetha = None
    clock.tick()
    if tipo == "GLOBAL_K":
        temp_knn = default_knn(train, min_k, max_k)
        traintime = clock.tock()
        clock.tick()
        score, k = temp_knn.get_score(test)
        testtime = clock.tock()
    elif tipo == "LOCAL_BEST_K":
        temp_knn = local_best_k(train, min_k, max_k)
        traintime = clock.tock()
        clock.tick()
        score, k = temp_knn.classify_set(test)
        testtime = clock.tock()
    elif tipo == "WEIGHTED_HARMONIC":
        temp_knn = mlmkhnn(train, min_k, max_k)
        traintime = clock.tock()
        clock.tick()
        score, k = temp_knn.get_precision(test)
        testtime = clock.tock()
    elif tipo == "FUZZY_HUBNESS":
        temp_knn = fuzzyhubness(train, min_k, max_k)
        traintime = clock.tock()
        clock.tick()
        score, k, thetha = temp_knn.predict_set(test)
        testtime = clock.tock()
    return score, traintime, testtime, k, thetha


def get_results(tipo, precision, train_time, test_time):
    return tipo + ": " + precisions_to_string(precision) \
           + "(Train: " + times_to_string(train_time) + ")" \
           + "(Test: " + times_to_string(test_time) + ")"


def times_to_string(times):
    return str(round(sum(times) / len(times), 6) * 1000) + "~" + str(
        round(float(np.std(np.array(times))), 6) * 1000) + " ms"


def precisions_to_string(precisions):
    return str(100 * round(sum(precisions) / len(precisions), 4)) + "~" + str(
        round(float(np.std(np.array(precisions))), 4) * 100)


def k_to_string(modelo, k_list):
    if modelo == "LOCAL_BEST_K":
        all_k = []
        for k in k_list:
            for i in k:
                all_k.append(i)
        return str(round(sum(all_k) / len(all_k), 4)) + "~" + str(
            round(float(np.std(np.array(all_k))), 4))
    else:
        return str(round(sum(k_list) / len(k_list), 4)) + "~" + str(
            round(float(np.std(np.array(k_list))), 4))


def thetha_to_string(thethalist):
    return str(round(sum(thethalist) / len(thethalist), 4)) + "~" + str(
        round(float(np.std(np.array(thethalist))), 4))


if __name__ == "__main__":
    main()
