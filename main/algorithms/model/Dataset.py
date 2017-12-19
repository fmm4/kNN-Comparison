import numpy as np

from Sample import Sample
from random import shuffle


class Dataset:
    samples = []
    nome = ""

    def __init__(self, nome, samples):
        self.samples = samples
        self.nome = nome

    def getNome(self):
        return self.nome

    def setNome(self,nome):
        self.nome = nome

    def insert(self, new):
        if not isinstance(new, Sample):
            print("Can only add samples in a dataset")
            return
        elif len(self.samples) != 0:
            if self.samples[0].same_type(new):
                self.samples.append(new)
            else:
                print("Failed to add")
        else:
            self.samples.append(new)

    def insert_all(self, new):
        for k in new:
            self.samples.append(Sample(k, "b"))

    def get_X(self):
        return np.array([k.parameters for k in self.samples])

    def get_y(self):
        return np.array([k.sampleClass for k in self.samples])

    def n_folds(self, n):
        new = self.samples
        shuffle(new)
        return np.array_split(np.array(new), n)

    def get_info(self):
        nrOfSamples = len(self.samples)
        nrOfClass = len(np.unique(np.array([d.sampleClass for d in self.samples])))
        nrOfParamet =  len(self.samples[0].parameters)
        return {"samples": nrOfSamples, "classes":nrOfClass, "parameters":nrOfParamet}

    def __repr__(self):
        return ("~Dataset ["+str([k for k in self.samples])+"]")

