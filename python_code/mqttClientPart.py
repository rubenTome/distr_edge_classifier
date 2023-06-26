import time
iniTime = time.time()
import paho.mqtt.client as mqtt
import partitionfunctions_python as partf
import fine_analysis_python as finean
import sys
import socket
from numpy import unique
from random import randint

#cliente para crear y publicar las particiones, y recibir resultados
#ARRANCAR DESPUES DE LOS CLASIFICADORES 

#PARAMETROS 

#number of partitions
Pset = sys.argv[1].replace(" ", "").strip("][").split(",")
for i in range(len(Pset)):
    Pset[i] = int(Pset[i])

#array de weighed belief values
wbelief = {i:[] for i in Pset}
#TODO con classesDist no tiene mejores metricas que la version aleatoria
classesDist = []#[[0, 1, 7], [3, 4, 5, 8], [2, 6, 9]]

clasTime = {i:0 for i in Pset}

NTRAIN = int(sys.argv[2])
NTEST = int(sys.argv[3])
if (sys.argv[4] == "balanced"):
    is_balanced = True
elif (sys.argv[4] == "unbalanced"):
    is_balanced = False
else:
    exit("ERROR: invalid argument ", sys.argv[4])

#posibles valores: "pnw", "piw", "unw", "random"
weighingStrategy = sys.argv[5]

dataset = sys.argv[6]

#TODO revisar, fallo en rbind (rpy2)
if(len(sys.argv) == 8):
    datasetTrain = sys.argv[7]
else:
    datasetTrain = ""

BROKER_IP = socket.gethostbyname(socket.gethostname())

#CREACION PARTICIONES

def dataframeToStr(df):
    #nombres de las columnas de cada particion
    string = ",".join(df.columns) + "\n"
    for k in range(len(df.values)):
        #valores de las filas de cada particion
        string = string + ",".join([str(l) for l in df.values[k]]) + "\n"
    return string

def strToArray(str):
    splitedStr = str.split("\n")
    resArr = [[] for _ in range(len(splitedStr))]
    for i in range(len(splitedStr)):
        splitedStr[i] = splitedStr[i][1:-1].split(",")
        for j in range(len(splitedStr[i])):
            if(len(splitedStr[i][j]) == 0):
                continue
            else:
                if (splitedStr[i][j] == "0."):
                    splitedStr[i][j] == "0.0"
                resArr[i].append(float(splitedStr[i][j]))
    return resArr

def create_partitions():
    ds = partf.load_dataset(dataset, NTRAIN, NTEST, datasetTrain)
    global testClasses
    testClasses = ds["testclasses"]
    global uniqueClass
    uniqueClass = unique(testClasses)
    #creamos las particiones segun parametro classesDist e is_balanced
    if(len(classesDist) > 0):
        partitionFun = partf.create_selected_partition
    elif is_balanced:
        partitionFun = partf.create_random_partition
    else:
        partitionFun = partf.create_perturbated_partition
    partitions = [[] for _ in range(len(Pset))]
    test = ds["testset"]
    #creamos particiones para cada nodo
    if(len(classesDist) > 0):
        for p in range(len(Pset)):
            partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p], classesDist)
    else:
        for p in range(len(Pset)):
            partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p])
    return (partitions, test)

def distClass(usedClassifier, clasTime, secondTime):
    splitedName = dataset.split("/")
    dsName = splitedName[len(splitedName) - 1].split(".")[0]
    file = open("rdos_" + str(NTRAIN) + "_" + str(NTEST) + "_" + usedClassifier + "_" + weighingStrategy + "_" + dsName + "_distr.txt", "w")
    file.write("From dataset " + dataset + "\n")
    classArr = {i:[] for i in Pset}
    tempArr = []
    for i in (Pset):
        for j in range(len(wbelief[i][0])):
            for k in range(i):
                tempArr.append(wbelief[i][k][j])
            if (len(tempArr[0]) != 0):
                classArr[i].append(uniqueClass[finean.sum_rule(tempArr)])
            tempArr = []
    #tiempo en integrar los resultados de todos los clasificadores
    joiningTime = time.time() - secondTime
    for i in range(len(Pset)):
        file.write("\t" + str(Pset[i]) + " partitions:\n")
        file.write("\t" + str(classArr[Pset[i]]) + "\n")
        file.write("\taccuracy:\n\t" + str(finean.accu(classArr[Pset[i]], testClasses.tolist())) + "\n")
        file.write("\tprecision:\n\t" + str(finean.multi_precision(classArr[Pset[i]], testClasses.tolist())) + "\n")
        file.write("\trecall:\n\t" + str(finean.multi_recall(classArr[Pset[i]], testClasses.tolist())) + "\n")
        file.write("\texecution time:\n\t" + str(clasTime[Pset[i]] + joiningTime) + "\n\n")
    file.write("real values:\n\t" + str(testClasses.tolist()))
    client.publish("exit", 1, 2)

def listToStr(list):
    string = "["
    for i in range(len(list)):
        string += str(list[i])
        string += ","
    string += "]"
    return string

#MQTT
def on_connect(client, userdata, flags, rc):
    print("Connected partitions client with result code " + str(rc))
    client.unsubscribe("#")
    client.subscribe("results/#")
    client.subscribe("exit")
    #comprobamos que la weighting strategy sea correcta
    if (weighingStrategy != "pnw" and weighingStrategy != "piw" and weighingStrategy != "unw" and weighingStrategy != "random"):
        print("INVALID WEIGHING STRATEGY")
        client.publish("exit", 1, 2)
    if (weighingStrategy == "random"):
        randomMatrixStr = ""
        for i in range(NTEST):
            n = randint(0, 2)
            if (n == 0):
                randomMatrixStr += "1,0,0\n"
            if (n == 1):
                randomMatrixStr += "0,1,0\n"
            if (n == 2):
                randomMatrixStr += "0,0,1\n"
    else:
        randomMatrixStr = "not_random"
    partAndTest = create_partitions()
    uniqueClassStr = "[" + ",".join(str(i) for i in uniqueClass) + ",]"#TODO en strToList no necesitar la "," final
    for j in range(len(Pset)):
        for k in range(Pset[j]):
            message = dataframeToStr(partAndTest[0][j][k]) + "$" + weighingStrategy + "$" + dataframeToStr(partAndTest[1]) + "$" + uniqueClassStr + "$" + randomMatrixStr
            client.publish("partition/" + str(Pset[j]) + "." + str(k), message, 2)
            print("published partition " + str(Pset[j]) + "." + str(k))

def on_message(client, userdata, msg):
    if (msg.topic == "exit"):
        client.disconnect()
        exit()
    nPset = int(msg.topic.split("/")[1].split(".")[0])
    #en caso de un nPset > 1, el que mas tarda sobreescribe el valor
    clasTime[nPset] = time.time() - iniTime
    secondTime = time.time()
    splitedMsg = str(msg.payload)[2:-1].split("$")
    message = splitedMsg[0].replace("\\n", "\n")
    usedClassifier = splitedMsg[1]
    wbelief[nPset].append(strToArray(message))
    print("from topic " + msg.topic + ": received weighed belief values")
    allReceived = 1
    for i in Pset:
        if (len(wbelief[i]) != i):
            allReceived = 0
            break
    if (allReceived):
        print("all received")
        distClass(usedClassifier, clasTime, secondTime)

client = mqtt.Client("partitions_client", True)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, 1883)

client.loop_forever()