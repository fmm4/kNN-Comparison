from model.Dataset import Dataset
from model.Sample import Sample
import csv
import pandas as pd
from operator import itemgetter


def split_train_test(dataset, i):
    test = dataset[i]
    train_dataset = Dataset("Treino", array_of_samples_to_db([dataset[:i] + dataset[i + 1:]]))
    test_dataset = Dataset("Teste", test)
    return test_dataset, train_dataset


def array_of_samples_to_db(samples):
    valid = []
    for k in samples[0]:
        valid.extend(k)
    return valid


def sort_dict(dict, axis):
    list = sorted(dict.items(), key=itemgetter(axis))
    list.reverse()
    return list


def dbg(string):
    print str(string)


def read_database(name, index):
    array = []
    f = open("database/" + name + ".txt", "r")
    for line in f.read().splitlines():
        lista = line.split(",")
        classe = lista.pop(index)
        parameters = [float(i) for i in lista]
        array.append(Sample(parameters, classe))
    bd = Dataset(name, array)
    return bd


def read_landsat():
    array = []
    f = open("database/landsat.txt", "r")
    for line in f.read().splitlines():
        lista = line.split(" ")
        classe = lista.pop(len(lista) - 1)
        parameters = [float(i) for i in lista]
        array.append(Sample(parameters, classe))
    bd = Dataset("landsat", array)
    return bd


# def read_csv_database(name, index):
#     array = []
#     with open("database/" + name + ".csv", "r") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             values = row[0].split(";")
#             classe = lista.pop(index)
#             parameters = [float(i) for i in lista]
#             for k in row:
#                 print k.split(";")
# for line in f.read().splitlines():
#     lista = line.split(",")
#     classe = lista.pop(index)
#     parameters = [float(i) for i in lista]
#     array.append(Sample(parameters, classe))
# bd = Dataset(array)
# return bd

def read_xls_database(name):
    df = pd.read_excel("database/" + name + ".xls")
    array = []
    nparray = df[1:].as_matrix()
    for k in nparray:
        infos = k
        classe = infos[-1]
        parameters = [float(i) for i in infos[:-1]]
        array.append(Sample(parameters, classe))
    data_base = Dataset(name, array)
    return data_base

def read_yeast():
    array = []
    f = open("database/yeast.data", "r")
    for line in f.read().splitlines():
        lista = line.split()
        classe = lista.pop(len(lista) - 1)
        lista.pop(0)
        parameters = [float(i) for i in lista]
        array.append(Sample(parameters, classe))
    bd = Dataset("yeast", array)
    return bd

def read_ionosphere():
    array = []
    f = open("database/ionosphere.txt", "r")
    for line in f.read().splitlines():
        lista = line.split(",")
        classe = lista.pop(len(lista) - 1)
        parameters = [float(i) for i in lista]
        array.append(Sample(parameters, classe))
    bd = Dataset("ionosphere", array)
    return bd

def read_glass():
    array = []
    f = open("database/glass.txt", "r")
    for line in f.read().splitlines():
        lista = line.split(",")
        lista.pop(0)
        classe = lista.pop(len(lista) - 1)
        parameters = [float(i) for i in lista]
        array.append(Sample(parameters, classe))
    bd = Dataset("glass", array)
    return bd
    # array = []
    # with open(, "r") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         values = row[0].split(;)
    #         classe = lista.pop(index)
    #         parameters = [float(i) for i in lista]
    #         for k in row:
    #             print k.split(";")
    # for line in f.read().splitlines():
    #     lista = line.split(",")
    #     classe = lista.pop(index)
    #     parameters = [float(i) for i in lista]
    #     array.append(Sample(parameters, classe))
    # bd = Dataset(array)
    # return bd
