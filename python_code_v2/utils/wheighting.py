from utils.distance import energyDistR
from dcor import energy_distance as energyDistP
import pandas as pd

#possible values: energyDistR, energyDistP
ENERGY_FUNC = energyDistR

#All arguments must be lists

def pnw(predicted, train, test):
    distance = (1 / ENERGY_FUNC(train, test))
    predicted = predicted * distance
    return predicted

def piw(predicted, train, test):
    for i in range(len(predicted)):
        predicted[i] = predicted[i] * (1 / ENERGY_FUNC(train, [test[i]]))
    return predicted

def random(predicted):
    raise NotImplementedError