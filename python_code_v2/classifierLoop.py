import os
import sys

nodeTopic = sys.argv[1]
brokerIp = sys.argv[2]
models = ["knn",
		  "rf",
		  "svm",
		  "xgb"]
weightings = ["now",
			  "pnw",
			  "piwm",
			  "piw"]
#total executions per each configuration
nReps = 10
#number of datasets processed
nDatasets = 3

#80 executions per dataset with this parameters

for _ in range(nDatasets):
	for model in models:
		for weighting in weightings:
			for _ in range(int(nReps)):
				statement = "python3 classifierNode.py " + nodeTopic + " " + model + " " + brokerIp + " " + weighting
				print("loop:", statement)
				os.system(statement)
