import pandas as pd
import random as rd
import math as mt

def load_dataset(path, size):
    df = pd.read_csv(filepath_or_buffer=path, nrows=size)
    return df

def create_random_partition(data, nNodes, seed, trainSize=0.7, testSize=0.3):
    if trainSize + testSize != 1:
        raise ValueError("trainSize + testSize must be equal to 1")
    rd.seed(seed)
    #divide data in train and test
    n = len(data)
    trainN = mt.trunc(trainSize * n)
    testN = mt.trunc(testSize * n)
    trainSet = pd.DataFrame(columns=data.columns)
    testSet = pd.DataFrame(columns=data.columns)
    trainRows = []
    for _ in range(trainN):
        rdInt = rd.randint(0, n - 1)
        trainRows.append(rdInt)
        trainSet = pd.concat([trainSet, data.iloc[[rdInt]]])
    for _ in range(testN):
        rdInt = rd.randint(0, n - 1)
        while rdInt in trainRows:
            rdInt = rd.randint(0, n - 1)
        testSet = pd.concat([testSet, data.iloc[[rdInt]]])

    #divide train in nNodes
    nodeTrainN = mt.trunc(trainN / nNodes)
    nodeTrainSets = [pd.DataFrame(columns=data.columns) for _ in range(nNodes)]
    for i in range(nNodes):
        nodeTrainSets[i] = trainSet.iloc[i * nodeTrainN:(i + 1) * nodeTrainN]
    
    return nodeTrainSets, testSet

#calculamos distr clases original
#para cada nodo, multiplicamos la propr de clases entre 0.3 y 1.7 y normalizamos
#para cada nodo, cogemos el numero de patrones calculados anteriormente
def create_perturbated_partition(data, nNodes, seed, trainSize=0.7, testSize=0.3):
    if trainSize + testSize != 1:
        raise ValueError("trainSize + testSize must be equal to 1")
    rd.seed(seed)
    #divide data in train and test
    n = len(data)
    trainN = mt.trunc(trainSize * n)
    testN = mt.trunc(testSize * n)
    trainSet = pd.DataFrame(columns=data.columns)
    testSet = pd.DataFrame(columns=data.columns)
    #calculate original classes distribution

def create_selected_partition(data, nNodes, classesDist, trainSize=0.7, testSize=0.3):
    raise NotImplementedError