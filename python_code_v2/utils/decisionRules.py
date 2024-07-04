import numpy as np
import pandas as pd

# Decision rules

#We use only the best decision rule

#beliefs is a list of belief dataframes of each node
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
    