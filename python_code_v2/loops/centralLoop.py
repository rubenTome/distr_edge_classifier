import os
import sys
import time

nNodes = sys.argv[1]
train = "0.75"
test = "0.25"
nDatas = ["10000"]
partitions = ["perturbated", "random", "balanced", "selected"]
datasets = ["../datasets/reordered_mnist_train.csv",
            "../datasets/covtype.csv",
            "../datasets/connect-4Train.csv",
            "../datasets/HIGGS.csv"]
selectedDsFile = {"../datasets/reordered_mnist_train.csv":"conf_files/sel_part_conf_mnist.txt", 
                  "../datasets/covtype.csv":"conf_files/sel_part_conf_covtype.txt", 
                  "../datasets/connect-4Train.csv":"conf_files/sel_part_conf_connect4.txt"}
#decision rule to merge results
decisionRule = "sum"
#total executions per each configuration
nReps = 120
#number of executions with the same train and test data
DsReps = 1
for nData in nDatas:
    for partition in partitions:
        for dataset in datasets:
            for _ in range(nReps):
                for repConf in range(DsReps):
                    if DsReps == 1:
                        statement = "python3 centralNode.py " + nNodes + " " + nData + " " + train + " " + test + " " + partition + " " + dataset + " " + decisionRule + " -1"
                    else:
                        statement = "python3 centralNode.py " + nNodes + " " + nData + " " + train + " " + test + " " + partition + " " + dataset + " " + decisionRule + " " + str(repConf)
                    if partition == "selected":
                        if dataset == "../datasets/HIGGS.csv":
                            continue
                        else:
                            statement += " " + selectedDsFile[dataset]
                    print("loop:", statement)
                    print("rep:", repConf)
                    os.system(statement)
                    time.sleep(10)
