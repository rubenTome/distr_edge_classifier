from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def knn(train, test):
    raise NotImplementedError

def xgb(train, test):
    raise NotImplementedError

def rf(train, test):
    raise NotImplementedError

def svm(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    svc.fit(train, trainClasses)
    return svc.predict_proba(test[:].values)