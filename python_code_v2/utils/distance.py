import rpy2.robjects as ro
from numpy import array, asarray, pad
from scipy.spatial.distance import canberra, braycurtis

#energy distance from R, must intsall energy package in R (install.packages("energy"))
energy_r = ro.r('''
    energy_r = function (X , Y ) {
    X = as.matrix(X)
    Y = as.matrix(Y)
    energy::eqdist.e(rbind(X,Y), c(nrow(X), nrow(Y))) / var(as.vector(rbind(X,Y)))
    }
''')

#argumnets must be lists
def energyDistR(x, y):
    x = array(x)
    y = array(y)
    return float(asarray(energy_r(ro.r.matrix(ro.FloatVector(x.flatten(order="F")), nrow=x.shape[0]),
                                  ro.r.matrix(ro.FloatVector(y.flatten(order="F")), nrow=y.shape[0]))))

def canberraDist(x, y):
    x = array(x).flatten()
    y = array(y).flatten()
    if (len(x) > len(y)):
        y = pad(y, (0, len(x) - len(y)), "mean")
    elif (len(y) > len(x)):
        x = pad(x, (0, len(y) - len(x)), "mean")
    return canberra(x, y)

def braycurtisDist(x, y):
    x = array(x).flatten()
    y = array(y).flatten()
    if (len(x) > len(y)):
        y = pad(y, (0, len(x) - len(y)), "mean")
    elif (len(y) > len(x)):
        x = pad(x, (0, len(y) - len(x)), "mean")
    return braycurtis(x, y)