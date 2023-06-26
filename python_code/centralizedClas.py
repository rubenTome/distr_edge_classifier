import time
iniTime = time.time()
import partitionfunctions_python as partf
import fine_analysis_python as finean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from numpy import shape, arange, unique, argmax

print("CENTRALIZED CLASSIFIER")
TRAINFILE = "/home/ruben/FIC/Q8/TFG/clean_partition/scenariosimul/scenariosimulC8D5G3STDEV0.15.csv"
TESTFILE = ""
NTRAIN = 3000
NTEST = 500
classifier = "rf"

print("\t*Creating dataset")
ds = partf.load_dataset(TRAINFILE, NTRAIN, NTEST, TESTFILE)
#TODO func. particion por parametro
partitions = partf.create_random_partition(ds["trainset"], ds["trainclasses"], 1)
testset = ds["testset"]
testclasses = ds["testclasses"]
uniqueclasses = unique(testclasses)

partition = partitions[0].to_numpy()
nVars = shape(partition)[1] - 1
trainset = partition[:, arange(nVars)]
trainclasses = partition[:,[nVars]].flatten()
if (classifier == "rf"):
    cls = RandomForestClassifier(verbose=1)
elif (classifier == "knn"):
    cls = KNeighborsClassifier(n_neighbors = 2)
elif(classifier == "xgb"):
    cls = GradientBoostingClassifier(verbose=1)
else:
    exit("ERROR: invalid classifier " + classifier)
cls.fit(trainset, trainclasses)
print("\t*Calculating class probabilities")
testClass = cls.predict_proba(testset[:].values)

classResults = [-1 for _ in range(len(testClass))]
for i in range(len(testClass)):
    classResults[i] = uniqueclasses[argmax(testClass[i], axis=0)]

splitedName = TRAINFILE.split("/")
trainfilename = splitedName[len(splitedName) - 1].split(".")[0]

print("\t*Writing results")
file = open("rdos_" + "centralized_" + classifier + "_" + str(NTRAIN) + "_" + str(NTEST) + "_" + trainfilename + ".txt" , "w")
file.write("Centralized classifier results\n")
file.write("\taccuracy:\n\t" + str(finean.accu(classResults, testclasses.tolist())) + "\n")
file.write("\tprecision:\n\t" + str(finean.multi_precision(classResults, testclasses.tolist())) + "\n")
file.write("\trecall:\n\t" + str(finean.multi_recall(classResults, testclasses.tolist())) + "\n")
file.write("\texecution time:\n\t" + str(time.time() - iniTime) + "\n\n")