from utils.distance import energyDistR, canberraDist
#from dcor import energy_distance as energyDistP
import pandas as pd
import numpy as np

#possible values: energyDistR, energyDistP, canberraDist
DISTANCE_FUNC = canberraDist

#All arguments must be lists

def pnw(predicted, train, test):
    distance = (1 / DISTANCE_FUNC(train, test))
    predicted = predicted * distance
    return predicted

def piw(predicted, train, test):
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (1 / DISTANCE_FUNC(train, [test[i]]))
    return predicted

def piwm(predicted, train, test):
    trainMean = np.mean(np.array(train), axis=0)
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (1 / DISTANCE_FUNC(trainMean, test[i]))
    return predicted

def random(predicted):
    raise NotImplementedError