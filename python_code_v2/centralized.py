import time
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import utils.classifiers as classifiers
import pandas as pd
import numpy as np

#calculate accuracy, precision and recall
def computeMetrics(predicted, real):
    accuracy = metrics.accu(predicted, real)
    precision = metrics.multi_precision(predicted, real)
    recall = metrics.multi_recall(predicted, real)
    return accuracy, precision, recall

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
acc = []
prec = []
rec = []
t = []
resultsFile = open("results_centralized.txt", "a")
for path in paths:
    for classifier in classifiersL:
        for size in sizes:
            for i in range(reps):
                print("rep", i)
                timer = time.time()
                print("\tloading data...")
                data = data_loaders.load_dataset(path, size)
                #load data as it exists only 1 node
                trainData, testData = data_loaders.create_random_partition(data, 1, seed, trainSize, testSize)
                print("\ttraining...")
                #classify data and select the classes with highest belief value in each row
                results = pd.DataFrame(classifier(trainData[0], testData)).idxmax(axis=1).tolist()
                print("\tcomputing metrics...")
                testClasses = testData.iloc[:,-1:].to_numpy().flatten()
                testClasses = np.array(testClasses) - min(testClasses)
                metricsRes = computeMetrics(np.array(results), testClasses)
                acc.append(metricsRes[0])
                prec.append(metricsRes[1])
                rec.append(metricsRes[2])
                t.append(time.time() - timer)
            resultsFile.write("for dataset: " + path + " with size: " + str(size) + " and classifier: " + classNames[classifier] + "\n")
            resultsFile.write("mean accuracy: " + str(sum(acc) / len(acc)) + "\n")
            resultsFile.write("mean precision: " + str(sum(prec) / len(prec)) + "\n")
            resultsFile.write("mean recall: " + str(sum(rec) / len(rec)) + "\n")
            resultsFile.write("mean execution time: " + str(sum(t) / len(t)) + "\n")
            resultsFile.write("----------\n")
            acc = []
            prec = []
            rec = []
            t = []