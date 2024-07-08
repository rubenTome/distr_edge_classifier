from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def knn(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    model = KNeighborsClassifier()
    model.fit(train.values, trainClasses)
    return model.predict_proba(test[:].values)

def xgb(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    model = GradientBoostingClassifier()
    model.fit(train.values, trainClasses)
    return model.predict_proba(test[:].values)

def rf(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    model = RandomForestClassifier()
    model.fit(train.values, trainClasses)
    return model.predict_proba(test[:].values)

def svm(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    svc.fit(train.values, trainClasses)
    return svc.predict_proba(test[:].values)