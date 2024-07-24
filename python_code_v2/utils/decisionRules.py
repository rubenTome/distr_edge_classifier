import numpy as np
import pandas as pd

# Decision rules

#beliefs is a list of belief dataframes of each node

#add all the predicted beliefs values, choosing the highest
def sum_rule(beliefs):
    #add all belief values
    for i in range(len(beliefs) - 1):
        beliefs[0].add(beliefs[i + 1], fill_value=0)
    #select the class with the highest belief
    result = beliefs[0].idxmax(axis=1)
    #return the result as a list
    resultList = result.tolist()
    resultList = [int(i) for i in resultList]
    return resultList

#choose the highest value
def max_rule(beliefs):
    #each element in beliefs is a list of beliefs
    beliefsL = [i.to_numpy() for i in beliefs]
    resultList = []
    for i in range(len(beliefsL[0])):
        #get all beliefs for sample i
        sampleBeliefs = [beliefsL[j][i] for j in range(len(beliefsL))]
        #get index of max belief value of each node
        maxSampleIndex = np.argmax(sampleBeliefs, axis=1)
        #get the index of max value between nodes
        maxVal = -1
        maxIndex = -1
        for k in range(len(sampleBeliefs)):
            if sampleBeliefs[k][maxSampleIndex[k]] > maxVal:
                maxVal = sampleBeliefs[k][maxSampleIndex[k]]
                maxIndex = maxSampleIndex[k]
        resultList.append(maxIndex)
        maxVal = -1
        maxIndex = -1
    return resultList