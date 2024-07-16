import time
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import utils.classifiers as classifiers
import pandas as pd
import numpy as np

#calculate accuracy, precision and recall
def computeMetrics(predicted, real):
    acuracy = metrics.accu(predicted, real)
    precision = metrics.multi_precision(predicted, real)
    recall = metrics.multi_recall(predicted, real)
    return acuracy, precision, recall

reps = 10
paths = ["../datasets/covtype.csv", 
         "../datasets/HIGGS.csv", 
         "../datasets/connect-4Train.csv", 
         "../datasets/reordered_mnist_train.csv"]
sizes = [3500]
trainSize = 0.75
testSize = 0.25
seed = time.time()
classifiersL = [classifiers.knn, 
                classifiers.rf, 
                classifiers.svm, 
                classifiers.xgb
]
classNames = {classifiers.knn: "knn",
              classifiers.rf: "rf",
              classifiers.svm: "svm",
              classifiers.xgb: "xgb"
}
resultsFile = open("results_centralized.txt", "a")
for _ in range(reps):
    for path in paths:
        for size in sizes:
            for classifier in classifiersL:
                timer = time.time()
                print("loading data...")
                data = data_loaders.load_dataset(path, size)
                #load data as it exists only 1 node
                trainData, testData = data_loaders.create_random_partition(data, 1, seed, trainSize, testSize)
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