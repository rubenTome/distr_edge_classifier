import sklearn.neighbors as skn
import partitionfunctions_python as partf
import fine_analisis_python as fan
import numpy as np
import pandas as pd
import csv

# FUNCION CREATE_PERTURBATED_PARTITION NO FUNCIONA EN DATASETS DE 8 CLASES
# REVISAR FUNCION CREATE_RANDOM_PARTITION 

#CLASIFICADORES

def knn(partition):#partition es un pandas.DataFrame
    partition = partition.to_numpy()
    nVars = np.shape(partition)[1] - 1
    trainset = partition[:, np.arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    clf = skn.KNeighborsClassifier(n_neighbors = 2)
    clf.fit(trainset, trainclasses)

    return clf.score(ds["testset"].to_numpy(), ds["testclasses"].to_numpy().flatten())


#PARAMETROS 

totalresults = None

#how many reps per experiment
NREP = 5
#size of the total dataset (subsampple)
NSET = 100
#size of the train set, thse size of the test set will be NSET - NTRAIN
NTRAIN = 50

#number of partitions
Pset = [4]

is_balanced = False

datasets = ["../scenariosimul/scenariosimulC2D2G3STDEV0.15.csv", "../scenariosimul/scenariosimulC8D3G3STDEV0.05.csv"]

#some datasets are split into train and test, because of concept drift
testdatasets= [""]

classifiers = [knn]

#names for printing them
namesclassifiers = ["KNN"] 


#SIMULACION
for d in range(len(datasets)):
    
    ds = partf.load_dataset(datasets[d], NSET, NSET - NTRAIN)

    #creamos las particiones segun parametro is_balanced
    if is_balanced:
        partitionFun = partf.create_random_partition
    else:
        partitionFun = partf.create_perturbated_partition
    partitions = [[] for _ in range(len(Pset))]
    for p in range(len(Pset)):
            partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p])

    #creamos los clasificadores con cada una de las particiones
    ClassifAcc = {}
    for ps in range(len(Pset)):
        for c in range(len(classifiers)):
            for pa in range(len(partitions[ps])):
                ClassifAcc[namesclassifiers[c] + "_" + str(Pset[ps]) + "_" + str(pa)] = (
                classifiers[c](partitions[ps][pa]))

    splitedPath = datasets[d].split("/")
    name = splitedPath[len(splitedPath) - 1]
    file = open("rdos_" + name, "w")
    writer = csv.writer(file)
    writer.writerow(["classifier", "npartitions", "accurancy"])
    for t in range(len(ClassifAcc)):
        label = list(ClassifAcc)[t]
        npart = label.split("_")[1]
        writer.writerow([namesclassifiers[0], npart, ClassifAcc[label]])