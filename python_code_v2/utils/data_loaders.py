import pandas as pd
import random as rd
import math as mt
import numpy as np

#REPITEN VALORES ENTRE TRAINSETS EN BALANCED Y PERTURBATED

def load_dataset(path, size):
    df = pd.read_csv(filepath_or_buffer=path, nrows=size)
    return df

#get random samples from data and divide them in nNodes
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
    #generate n random numbers
    randomN = rd.sample(range(n), n)
    #choose random numbers as index for train and test samples
    for _ in range(trainN):
        trainSet = pd.concat([trainSet, data.iloc[[randomN.pop()]]])
    for _ in range(testN):
        testSet = pd.concat([testSet, data.iloc[[randomN.pop()]]])

    #divide train in nNodes
    nodeTrainN = mt.trunc(trainN / nNodes)
    nodeTrainSets = [pd.DataFrame(columns=data.columns) for _ in range(nNodes)]
    for i in range(nNodes):
        nodeTrainSets[i] = trainSet.iloc[i * nodeTrainN:(i + 1) * nodeTrainN]
    
    return nodeTrainSets, testSet

#modify the original distribution multiplying it by a random number between 0.25 and 1.75
def create_perturbated_partition(data, nNodes, seed, trainSize=0.7, testSize=0.3):
    if trainSize + testSize != 1:
        raise ValueError("trainSize + testSize must be equal to 1")
    rd.seed(seed)
    #divide data in train and test
    n = len(data)
    trainN = mt.trunc(trainSize * n)
    testN = mt.trunc(testSize * n)
    testSet = pd.DataFrame(columns=data.columns)
    #calculate original classes distribution
    classesDist = data["classes"].value_counts() / sum(data["classes"].value_counts())
    classesDist.sort_index(inplace=True)
    classes = classesDist.index.tolist()
    classesDist = classesDist.tolist()
    print("original classes distribution:")
    print(classesDist)
    nodeTrainSets = [pd.DataFrame(columns=data.columns) for _ in range(nNodes)]
    for i in range(nNodes):
        #for each node, multiply class distribution between 0.3 and 1.7 and normalize
        pertClassesDist = [i * rd.uniform(0.25, 1.75) for i in classesDist]
        sumPertClassesDist = sum(pertClassesDist)
        pertClassesDist = [i / sumPertClassesDist for i in pertClassesDist]
        print("perturbed classes distribution for node " + str(i) + ":")
        print(pertClassesDist)
        #for each node, take the number of samples for each class calculated before
        pertNTrain = [mt.trunc(i * (trainN / nNodes)) for i in pertClassesDist]
        #train: for each class
        for j in range(len(pertNTrain)):
            #get all samples from class classes[j]
            dataClass = data[data["classes"] == classes[j]]
            #generate len(dataClass) random numbers
            randomNDataClass = rd.sample(range(len(dataClass)), len(dataClass))
            for _ in range(pertNTrain[j]):#popear as mostras de data, evitamos asi tamen o uso de trainrows
                #take one random number
                popIndex = randomNDataClass.pop()
                nodeTrainSets[i] = pd.concat([nodeTrainSets[i], 
                    dataClass.iloc[[popIndex]]])
                data.drop(dataClass.iloc[[popIndex]].index, inplace=True)
    #test: same as in random partition
    randomNtest = rd.sample(range(len(data)), len(data))
    for _ in range(testN):
        testSet = pd.concat([testSet, data.iloc[[randomNtest.pop()]]])
    nodeTrainSets[0].to_csv("train_set_1.csv")
    nodeTrainSets[1].to_csv("train_set_2.csv")
    nodeTrainSets[2].to_csv("train_set_3.csv")
    testSet.to_csv("test_set.csv")
    return nodeTrainSets, testSet

#create nNodes balanced partitions
def create_balanced_partition(data, nNodes, seed, trainSize=0.7, testSize=0.3):
    if trainSize + testSize != 1:
        raise ValueError("trainSize + testSize must be equal to 1")
    rd.seed(seed)
    #divide data in train and test
    n = len(data)
    testN = mt.trunc(testSize * n)
    testSet = pd.DataFrame(columns=data.columns)
    nodeTrainSets = [pd.DataFrame(columns=data.columns) for _ in range(nNodes)]
    #calculate number of train samples for each class (classesDist)
    classesDist = data["classes"].value_counts()
    classesDist.sort_index(inplace=True)
    classesDist = classesDist.to_numpy()
    #calculate number of samples for each class in each node
    nClassesNode = trainSize * classesDist / nNodes
    nClassesNode = nClassesNode.astype(int)
    #calculate possible class values
    classes = np.unique(data.iloc[:,-1:].to_numpy().flatten())
    #train
    #for each node
    for i in range(nNodes):
        #for each class
        for j in range(len(nClassesNode)):
            #get nClassesNode[j] samples from class j
            selClassSamples = data.loc[data['classes'] == classes[j]]
            nRandomNSel = rd.sample(range(len(selClassSamples)), len(selClassSamples))
            for _ in range(nClassesNode[j]):
                popIndex = nRandomNSel.pop()
                nodeTrainSets[i] = pd.concat([nodeTrainSets[i], 
                    selClassSamples.iloc[[popIndex]]])
                data.drop(selClassSamples.iloc[[popIndex]].index, inplace=True)
    #test: same as in random partition
    randomNtest = rd.sample(range(len(data)), len(data))
    for _ in range(testN):
        testSet = pd.concat([testSet, data.iloc[[randomNtest.pop()]]])
    return nodeTrainSets, testSet

def create_selected_partition(data, nNodes, classesDist, trainSize=0.7, testSize=0.3):
    raise NotImplementedError