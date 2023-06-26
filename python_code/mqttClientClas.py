import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from numpy import shape, arange, unique
from pandas import read_csv
from io import StringIO
import sys
import partitionfunctions_python as partf

#los clientes se subscriben a su particion y publican los resultados
#ARRANCAR PRIMERO LOS CLASIFICADORES

#CLASIFICADORES

def knn(partition, test):#partition es un pandas.DataFrame
    print("starting knn classifier")
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(trainset, trainclasses)
    testClass = clf.predict_proba(test[:].values)
    return testClass

def rf(partition, test, uniqueClass):#uniqueClass son todas las clases que existen
    print("starting rf classifier")
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    #clases que existen en este nodo
    ourTrainclasses = unique(trainclasses)
    rfc = RandomForestClassifier(verbose=1)
    rfc.fit(trainset, trainclasses)
    testClass = rfc.predict_proba(test[:].values)
    result = [[0 for _ in range(len(uniqueClass))] for _ in range(len(testClass))]
    for i in range(len(testClass)):
        for j in range(len(testClass[i])):
            result[i][uniqueClass.index(float(ourTrainclasses[j]))] = testClass[i][j]
    return result

def xgb(partition, test):
    print("starting xgb classifier")
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    gbc = GradientBoostingClassifier(verbose=1)
    gbc.fit(trainset, trainclasses)
    testClass = gbc.predict_proba(test[:].values)
    return testClass

#PARAMETROS 

totalresults = None

#numero de cifras decimales
NDECIMALS = 2

#generar tablas con los resultados (+legible)
generateTables = True

CLASSIFIERID = sys.argv[1]
USEDCLASSIFIER = sys.argv[2]

#debe ser el nombre o ip del servidor mosquitto
BROKER_IP = sys.argv[3]

def strToList(string):
    list = []
    splitedList = string[1:-1].split(",")
    for i in range(len(splitedList) - 1):
        list.append(float(splitedList[i]))
    return list

def intStrToArray(str):
    splitedStr = str.split("\n")
    resArr = [[] for _ in range(len(splitedStr))]
    for i in range(len(splitedStr)):
        splitedStr[i] = splitedStr[i].split(",")
        for j in range(len(splitedStr[i])):
            if(len(splitedStr[i][j]) == 0):
                continue
            else:
                resArr[i].append(int(splitedStr[i][j]))
    return resArr

def extractData(message):
    message = message.replace("\\n", "\n")
    splitedMsg = message.split("$")
    partitions = read_csv(StringIO(splitedMsg[0][2:]))
    weighting = splitedMsg[1]
    test = read_csv(StringIO(splitedMsg[2]))
    uniqueClass = strToList(splitedMsg[3])
    if (splitedMsg[4][:-1] != "not_random"):
        randomMatrix = intStrToArray(splitedMsg[4][:-1])
    else:
        randomMatrix = "not_random"
    return partitions, weighting, test, uniqueClass, randomMatrix

#ENTRENAMIENTO Y CLASIFICACION

def classify(partition, weighting, test, uniqueClass, randomMatrix):
    #obtenemos distancia entre test y partition
    classes, counts = unique(partition["classes"], return_counts=True)
    print("training node with classes ", classes)
    print("counts per class: ", counts)
    partitionMatrix = partition.drop('classes', axis=1).values
    testMatrix = test.values
    #en caso de no pesar los resultados, continuamos sin hacer nada
    if (weighting == "piw"):
        print("calculating distances")
        inverseDistance = [0 for _ in range(len(testMatrix))]
        for i in range(len(testMatrix)):
            print("instance ", i, end="\r")
            inverseDistance[i] = 1 / partf.end_R([testMatrix[i]], partitionMatrix)
    elif(weighting == "pnw"):
        print("calculating distances")
        inverseDistance = 1 / partf.end_R(testMatrix, partitionMatrix)
    #obtenemos belief values
    if (USEDCLASSIFIER == "knn"):
        classifierOutput = knn(partition, test, uniqueClass)
    elif(USEDCLASSIFIER == "rf"):
        classifierOutput = rf(partition, test, uniqueClass)
    elif(USEDCLASSIFIER == "xgb"):
        classifierOutput = xgb(partition, test, uniqueClass)
    else:
        print("UNKNOWN CLASSIFIER")
        exit(0)
    #pesamos los belief values
    print("weighting strategy:", weighting)
    if (weighting == "piw"):
        for i in range(len(classifierOutput)):
            for j in range(len(classifierOutput[i])):
                classifierOutput[i][j] = classifierOutput[i][j] * inverseDistance[i]
    elif (weighting == "pnw"):
        for i in range(len(classifierOutput)):
            for j in range(len(classifierOutput[i])):
                classifierOutput[i][j] = classifierOutput[i][j] * inverseDistance
    elif(weighting == "random"):
        splitedId = CLASSIFIERID.split(".")
        classIndex = int(splitedId[len(splitedId) - 1])
        for i in range(len(classifierOutput)):
            for j in range(len(classifierOutput[i])):
                classifierOutput[i][j] = classifierOutput[i][j] * randomMatrix[i][classIndex]
    #consruimos la respuesta
    stringOutput = ""
    for i in range(len(classifierOutput)):
        stringOutput += "["
        for j in range(len(classifierOutput[i])):
            stringOutput += str(classifierOutput[i][j])
            stringOutput += ","
        stringOutput += "]\n"
    return stringOutput

#MQTT
#se llama al conectarse al broker
def on_connect(client, userdata, flags, rc):
    print("Connected classification client with result code " + str(rc))
    client.unsubscribe("#")
    #nos subscribimos a este tema
    client.subscribe("partition/" + CLASSIFIERID)
    client.subscribe("exit")
    print("\nSubscribed to partition/" + CLASSIFIERID)

#se llama al obtener un mensaje del broker
def on_message(client, userdata, msg):
    if (msg.topic == "exit"):
        client.disconnect()
        exit(0)
    partition, weighting, test, uniqueClass, randomMatrix = extractData(str(msg.payload))
    print("received", msg.topic)
    classifiedData = classify(partition, weighting, test, uniqueClass, randomMatrix)
    print("pubish weighed belief values:\n", classifiedData)
    client.publish("results/" + CLASSIFIERID, classifiedData + "$" + USEDCLASSIFIER, 2)

client = mqtt.Client("clas_client_" + CLASSIFIERID, True)
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect(BROKER_IP, 1883)

client.loop_forever()