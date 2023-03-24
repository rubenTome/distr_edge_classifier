import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def svm_classifier_prob(trainset, trainclasses, testset, testclasses):
    trainset = np.asarray(trainset)
    testset = np.asarray(testset)
    trainclasses = np.asarray(trainclasses)
    testclasses = np.asarray(testclasses)
    clf = svm.SVC(probability=True)
    clf.fit(trainset, trainclasses)
    prob = clf.predict_proba(testset)
    return {"prob": prob}

def forest_classifier_prob(trainset, trainclasses, testset, testclasses):
    trainset = np.asarray(trainset)
    testset = np.asarray(testset)
    trainclasses = np.asarray(trainclasses)
    testclasses = np.asarray(testclasses)
    clf = RandomForestClassifier()
    clf.fit(trainset, trainclasses)
    prob = clf.predict_proba(testset)
    return {"prob": prob}

def xgb_classifier_prob(trainset, trainclasses, testset, testclasses):
    trainset = pd.DataFrame(trainset)
    testset = pd.DataFrame(testset)
    trainclasses = np.asarray(trainclasses) - 1
    testclasses = np.asarray(testclasses) - 1
    clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_jobs=2,
                        min_child_weight=1, num_class=len(np.unique(np.concatenate((trainclasses, testclasses)))),
                        objective='multi:softprob')
    clf.fit(trainset, trainclasses)
    prob = clf.predict_proba(testset)
    return {"prob": prob}

def multinom_classifier_prob(trainset, trainclasses, testset, testclasses):
    trainset = pd.DataFrame(trainset)
    testset = pd.DataFrame(testset)
    trainclasses = np.asarray(trainclasses)
    testclasses = np.asarray(testclasses)
    clf = MLPClassifier()
    clf.fit(trainset, trainclasses)
    prob = clf.predict_proba(testset)
    if len(np.unique(trainclasses)) == 2:
        prob = np.array([1-p, p] for p in prob)
    return {"prob": prob}

def lda_classifier_prob(trainset, trainclasses, testset, testclasses):
    trainset = np.asarray(trainset) + np.random.normal(0, 0.001, len(trainset))

