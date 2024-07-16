import os
import sys
import time

nNodes = sys.argv[1]
train = "0.75"
test = "0.25"
nDatas = ["3500"]
partitions = ["random"]
datasets = ["../datasets/covtype.csv",
            "../datasets/HIGGS.csv",
            "../datasets/connect-4Train.csv",
            "../datasets/reordered_mnist_train.csv"]
#total executions per each configuration
nReps = 160
for nData in nDatas:
    for partition in partitions:
        for dataset in datasets:
            for _ in range(nReps):
                statement = "python3 centralNode.py " + nNodes + " " + nData + " " + train + " " + test + " " + partition + " " + dataset
                print("loop:", statement)
                os.system(statement)
                time.sleep(6)
