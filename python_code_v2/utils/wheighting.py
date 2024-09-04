import pandas as pd
import numpy as np

#x and y must be lists

def pnw(predicted, train, test, distance_func):
    distance = (1 / distance_func(train, test))
    predicted = predicted * distance
    return predicted

def piw(predicted, train, test, distance_func):
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (1 / distance_func(train, [test[i]]))
    return predicted

def piwm(predicted, train, test, distance_func):
    trainMean = np.mean(np.array(train), axis=0)
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (1 / distance_func(trainMean, test[i]))
    return predicted