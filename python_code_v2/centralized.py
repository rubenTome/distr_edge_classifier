import time
timer = time.time()
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import utils.classifiers as classifiers
import pandas as pd
import numpy as np
import io

#calculate accuracy, precision and recall
def computeMetrics(predicted, real):
    acuracy = metrics.accu(predicted, real)
    precision = metrics.multi_precision(predicted, real)
    recall = metrics.multi_recall(predicted, real)
    return acuracy, precision, recall

paths = ["../datasets/HIGGS.csv"]
sizes = [10, 100, 1000]
classifiersL = [classifiers.svm]
classNames = {classifiers.svm: "svm",
              classifiers.rf: "rf",
              classifiers.xgb: "xgb",
              classifiers.knn: "knn"
}
resultsFile = open("results.txt", "w")
for path in paths:
    for size in sizes:
        for classifier in classifiersL:
            print("loading data...")
            data = data_loaders.load_dataset(path, size)
            #load data as it exists only 1 node
            trainData, testData = data_loaders.create_random_partition(data, 1)
            print("training...")
            #classify data and select the classes with highest belief value in each row
            results = pd.DataFrame(classifier(trainData[0], testData)).idxmax(axis=1).tolist()
            print("computing metrics...")
            acc, prec, rec = computeMetrics(np.array(results), testData.iloc[:,-1:].to_numpy().flatten())
            execTime = time.time() - timer
            resultsFile.write("for dataset: " + path + " with size: " + str(size) + " and classifier: " + classNames[classifier] + "\n")
            resultsFile.write("accuracy: " + str(acc) + "\n")
            resultsFile.write("precision: " + str(prec) + "\n")
            resultsFile.write("recall: " + str(rec) + "\n")
            resultsFile.write("execution time: " + str(execTime) + "\n")
            resultsFile.write("----------\n")