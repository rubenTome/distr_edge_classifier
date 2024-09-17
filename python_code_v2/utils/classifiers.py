from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

def knn(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    trainClasses = trainClasses - min(test.classes)
    uniqueTrainClasses = np.sort(np.unique(trainClasses))
    uniqueTestClasses = np.sort(np.unique(test.classes))
    model = KNeighborsClassifier()
    model.fit(train.iloc[:, :-1].values, trainClasses)
    predicted = model.predict_proba(test.iloc[:, :-1].values)
    newPredicted = [[0.0 for _ in range(len(uniqueTestClasses))] for _ in range(len(predicted))]
    for i in range(len(newPredicted)):
        for j in range(len(newPredicted[i])):
            for k in range(len(uniqueTrainClasses)):
                if j == uniqueTrainClasses[k]:
                    newPredicted[i][j] = predicted[i][k]
    return np.array(newPredicted, dtype=float)

def xgb(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    trainClasses = trainClasses - min(test.classes)
    uniqueTrainClasses = np.sort(np.unique(trainClasses))
    uniqueTestClasses = np.sort(np.unique(test.classes))
    model = GradientBoostingClassifier()
    model.fit(train.iloc[:, :-1].values, trainClasses)
    predicted = model.predict_proba(test.iloc[:, :-1].values)
    newPredicted = [[0.0 for _ in range(len(uniqueTestClasses))] for _ in range(len(predicted))]
    for i in range(len(newPredicted)):
        for j in range(len(newPredicted[i])):
            for k in range(len(uniqueTrainClasses)):
                if j == uniqueTrainClasses[k]:
                    newPredicted[i][j] = predicted[i][k]
    return np.array(newPredicted, dtype=float)

def rf(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    trainClasses = trainClasses - min(test.classes)
    uniqueTrainClasses = np.sort(np.unique(trainClasses))
    uniqueTestClasses = np.sort(np.unique(test.classes))
    model = RandomForestClassifier()
    model.fit(train.iloc[:, :-1].values, trainClasses)
    predicted = model.predict_proba(test.iloc[:, :-1].values)
    newPredicted = [[0.0 for _ in range(len(uniqueTestClasses))] for _ in range(len(predicted))]
    for i in range(len(newPredicted)):
        for j in range(len(newPredicted[i])):
            for k in range(len(uniqueTrainClasses)):
                if j == uniqueTrainClasses[k]:
                    newPredicted[i][j] = predicted[i][k]
    return np.array(newPredicted, dtype=float)

def svm(train, test):
    trainClasses = np.array(train.iloc[:, -1:].values.tolist()).flatten()
    trainClasses = trainClasses - min(test.classes)
    uniqueTrainClasses = np.sort(np.unique(trainClasses))
    uniqueTestClasses = np.sort(np.unique(test.classes))
    svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    svc.fit(train.iloc[:, :-1].values, trainClasses)
    predicted = svc.predict_proba(test.iloc[:, :-1].values)
    newPredicted = [[0.0 for _ in range(len(uniqueTestClasses))] for _ in range(len(predicted))]
    for i in range(len(newPredicted)):
        for j in range(len(newPredicted[i])):
            for k in range(len(uniqueTrainClasses)):
                if j == uniqueTrainClasses[k]:
                    newPredicted[i][j] = predicted[i][k]
    return np.array(newPredicted, dtype=float)
