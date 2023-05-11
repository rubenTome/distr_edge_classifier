import rpy2.robjects as ro
import numpy as np
import numpy.random as nprd
import scipy.spatial as scs
import random
import pandas as pd
import sys
import math as mt
import statistics as st

# ditancia "energy" energy.stat
energy_r = ro.r('''
    energy_r = function (X , Y ) {
    X = as.matrix(X)
    Y = as.matrix(Y)
    energy::eqdist.e(rbind(X,Y), c(nrow(X), nrow(Y))) / var(as.vector(rbind(X,Y)))
    }
''')

#x e y en caso de usar la distancia de R deben pasarse como robjects
#PNW
def end(x, y):#x e y son del tipo list
    x = np.array(x)
    y = np.array(y)
    xR = ro.r.matrix(ro.FloatVector(x.flatten(order="F")), nrow=x.shape[0])
    yR = ro.r.matrix(ro.FloatVector(y.flatten(order="F")), nrow=y.shape[0])
    return float(np.asarray(energy_r(xR,yR)))

def sample_n_from_csv(filename:str, n:int=100, total_rows:int=None) -> pd.DataFrame:
    if total_rows==None:
        with open(filename,"r") as fh:
            total_rows = sum(1 for row in fh)
    if(n>total_rows):
        print("Error: n > total_rows", file=sys.stderr) 
    skip_rows =  random.sample(range(1,total_rows+1), total_rows-n)
    return pd.read_csv(filename, skiprows=skip_rows)

def load_dataset(filename, maxsize, trainsize, testfilename = ""):
    dataset = {
        "trainset": None,
        "trainclasses": None,
        "testset": None,
        "testclasses": None
    }

    samp = sample_n_from_csv(filename, maxsize).sample(frac = 1)
    sampShape = samp.shape
    #indices de filas en trainset y testset son secuenciales no aleatorios
    dataset["trainset"] = samp.iloc[:trainsize, np.arange(sampShape[1] - 1)]
    dataset["trainclasses"] = samp.iloc[:trainsize].loc[:, "classes"]

    if testfilename == "":
        dataset["testset"] = samp.iloc[trainsize:, np.arange(sampShape[1] - 1)]
    else:
        dataset["testset"] = sample_n_from_csv(filename, maxsize - trainsize)

    dataset["testclasses"] = samp.iloc[trainsize:].loc[:, "classes"]

    return dataset   

def create_random_partition(trainset, trainclasses, npartitions):
    classes = np.unique(trainclasses)
    classesLen = len(classes)
    joined = pd.concat([trainset, trainclasses.reindex(trainset.index)], axis=1)
    groups = joined.groupby(["classes"], group_keys=True).apply(lambda x: x)

    #groupsList es una lista con 1 dataframe por clase
    groupsList = [pd.DataFrame() for _ in range(classesLen)]
    for i in range(classesLen):
        groupsList[i] = groups.xs(i + 1, level = "classes")

    #groupListPart es una lista que subdivide cada dataframe de groupList npartition veces
    groupsListPart = [pd.DataFrame() for _ in range(classesLen * npartitions)]
    for i in range(classesLen):
        gListShape = groupsList[i].shape
        for j in range(npartitions):
            groupsListPart[i * npartitions + j] = groupsList[i].sample(
                                                    n = np.floor(gListShape[0] / npartitions).astype(int), replace = True)
    
    #partition es el resultado de create_random_partition() 
    partitions = [pd.DataFrame() for _ in range(npartitions)]
    for i in range(npartitions):
        partitions[i] = groupsListPart[i]
        for j in range(1, classesLen):
            partitions[i] = pd.concat([partitions[i].reset_index(drop = True), 
                                groupsListPart[i + j * npartitions]])
    
    return partitions

def tablef(list, trainclasses):
    classes = np.unique(trainclasses)
    table = [0 for _ in range(len(classes))]
    for i in range(len(classes)):
        for j in range(len(list)):
            if classes[i] == list[j]:
                table[i] += 1
    return table

def whichf(arr, n):
    indexes = []
    for i in range(len(arr)):
        if arr[i] == n:
            indexes.append(i)
    return indexes

def deleteRowsDf(dataframe, rows):
    rows.sort()
    rows = np.flip(rows)
    for i in range(len(rows)):
        dataframe.drop([dataframe.index[rows[i]]], inplace=True)
    return dataframe

def create_perturbated_partition(trainset, trainclasses, npartitions):
    listRes = [[] for _ in range(npartitions)]

    remainingset = pd.DataFrame(trainset)
    remainingclasses = np.array(trainclasses)
    C = len(np.unique(trainclasses))
    partitions = []
    partitionclasses = [[] for _ in range(npartitions - 1)]

    for i in range(npartitions-1):
        N = len(remainingclasses)
        P = npartitions - i
        prop = np.array(tablef(remainingclasses, trainclasses)) / N
        dev = prop * nprd.uniform(0.1, 0.9, C)
        dev = dev / np.sum(dev)
        
        if i == 0:
            dev = prop

        observations = np.floor(dev * (N / P))
        partitions.append(pd.DataFrame())
        
        for j in range(C):
            rem = whichf(remainingclasses, j + 1)

            if (len(rem) == 0):
                exit("ERROR NO ELEMENTS  OF CLASS " + str(j))

            nobs = observations[j]

            if ((nobs == [0]).all()):
                nobs = 1

            nremclass = len(rem) - 1 #menos uno ?
            nobs = int(min(nobs, nremclass))
            selectedobs = np.array(random.sample(rem, nobs))

            if (len(rem) == 1):
                selectedobs = rem

            partitions[i] = pd.concat([partitions[i], remainingset.iloc[selectedobs]], ignore_index = True)

            partitionclasses[i] = np.append(partitionclasses[i], remainingclasses[selectedobs]).astype("int")

            if((tablef(remainingclasses, trainclasses)[j] - nobs) < 1):
                toadd = nobs
                remainingset = pd.concat([remainingset, remainingset.iloc[rem[:toadd]]])
                remainingclasses = np.append(remainingclasses, [remainingclasses[i] for i in rem[:toadd]])  

            remainingset = deleteRowsDf(remainingset, selectedobs)
            remainingclasses = np.delete(remainingclasses, selectedobs)

    partitions.append(remainingset)
    partitionclasses.append(remainingclasses)
    for i in range(npartitions):        
        lenPartClass = len(partitionclasses[i])
        lenPart = partitions[i].shape[0]
        while (lenPart != lenPartClass):
            partitionclasses[i] = np.delete(partitionclasses[i], lenPartClass - 1)
        
        partitions[i]["classes"] = partitionclasses[i]
        listRes[i] = partitions[i]

    return listRes

def distancef(x, y):
    arrXY = pd.concat([x, y])
    distances = scs.distance_matrix(arrXY, arrXY)
    dshape = distances.shape
    for i in range(dshape[0]):
        for j in range(dshape[1]):
            distances[i][j] = mt.exp(- distances[i][j])
    return pd.DataFrame(distances)

#PIW
def energy_wheights_sets(trainset, testset, bound=4):
    result = {
        "weights": None,
        "val": None
    }
    n = trainset.shape[0]
    distances = distancef(trainset, testset)
    K = distances.iloc[0:n, 0:n]
    k = distances.iloc[0:n, n:n + testset.shape[0]]
    WB = distances.iloc[n:n + testset.shape[0], n:n + testset.shape[0]]
    k = k.mean(axis = 1)
    B = 0
    c = np.array(-k)
    H = K
    H =  H.to_numpy().flatten(order="F")
    A = np.zeros((n, n))
    for i in range(A.shape[1]):
        A[0][i] = 1
    A = A.flatten(order="F")
    b = np.zeros(n)
    r = np.ones(n)
    l = np.zeros(n)
    u = np.ones(n)
    bound = bound

    ipopf = ro.r('''
    ipopf = function (c, H, A, b, l, u, r, bound) {
        library(kernlab)
        primal(ipop(c, H, A, b, l, u, r, sigf=4, maxiter = 45, bound=bound, margin = 0.01, verb=FALSE))
    }''')

    result["weights"] = ipopf(ro.vectors.FloatVector(c),
                            ro.r.matrix(ro.FloatVector(H), nrow=n),
                            ro.r.matrix(ro.IntVector(A), nrow=n),
                            ro.vectors.IntVector(b),
                            ro.vectors.IntVector(l),
                            ro.vectors.IntVector(u),
                            ro.vectors.IntVector(r),
                            bound)
    result["val"] = (np.matmul(-2 * np.array(k), result["weights"]) + 
                     np.matmul(np.matmul(result["weights"], K), result["weights"]) + 
                     WB.stack().mean())

    return result

#no usada
def kfun(x, y):
    return -np.sum(np.power(np.subtract(x, y), 2)) + np.sum(np.power(x, 2)) + np.sum(np.power(y, 2))