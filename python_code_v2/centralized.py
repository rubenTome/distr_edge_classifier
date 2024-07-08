import time
timer = time.time()
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import utils.classifiers as classifiers
import pandas as pd
import numpy as np
import io

PATH = "../datasets/HIGGS.csv"
SIZE = 1000

#calculate accuracy, precision and recall
def computeMetrics(predicted, real):
    acuracy = metrics.accu(predicted, real)
    precision = metrics.multi_precision(predicted, real)
    recall = metrics.multi_recall(predicted, real)
    return acuracy, precision, recall

print("loading data...")
data = data_loaders.load_dataset(PATH, SIZE)
#load data as it exists only 1 node
trainData, testData = data_loaders.create_random_partition(data, 1)
print("training...")
#classify data and select the classes with highest belief value in each row
results = pd.DataFrame(classifiers.svm(trainData[0], testData)).idxmax(axis=1).tolist()
print("computing metrics...")
acc, prec, rec = computeMetrics(np.array(results), testData.iloc[:,-1:].to_numpy().flatten())
execTime = time.time() - timer
print("accuracy:", acc)
print("precision:", prec)
print("recall:", rec)
print("execution time:", execTime)