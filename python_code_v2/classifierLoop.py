import os
import sys

nodeTopic = sys.argv[1]
brokerIp = sys.argv[2]
models = ["knn", "rf", "svm", "xgb"]
weightings = ["piw", "pnw"]
#total executions per each configuration
nReps = 10

#80 executions with this parameters

for model in models:
    for weighting in weightings:
        for _ in range(int(nReps)):
            statement = "python3 classifierNode.py " + nodeTopic + " " + model + " " + brokerIp + " " + weighting
            print("loop:", statement)
            os.system(statement)
