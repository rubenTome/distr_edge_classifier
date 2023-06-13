import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from numpy import shape, arange
from pandas import read_csv
from io import StringIO
import sys
import partitionfunctions_python as partf

#los clientes se subscriben a su particion y publican los resultados
#ARRANCAR PRIMERO LOS CLASIFICADORES

#CLASIFICADORES

def knn(partition, test):#partition es un pandas.DataFrame
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(trainset, trainclasses)
    testClass = clf.predict_proba(test[:].values)
    return testClass

def rf(partition, test):
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    rfc = RandomForestClassifier()
    rfc.fit(trainset, trainclasses)
    testClass = rfc.predict_proba(test[:].values)
    return testClass

def xgb(partition, test):
    partition = partition.to_numpy()
    nVars = shape(partition)[1] - 1
    trainset = partition[:, arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    gbc = GradientBoostingClassifier()
    gbc.fit(trainset, trainclasses)
    testClass = gbc.predict_proba(test[:].values)
    return testClass

#PARAMETROS 

totalresults = None

#numero de cifras decimales
NDECIMALS = 2

#generar tablas con los resultados (+legible)
generateTables = True

#debe ser el nombre o ip

CLASSIFIERID = sys.argv[1]
USEDCLASSIFIER = sys.argv[2]

BROKER_IP = sys.argv[3]

def strToList(string):
    list = []
    splitedList = string[1:-1].split(",")
    for i in range(len(splitedList) - 1):
        list.append(float(splitedList[i]))
    return list

def extractData(message):
    message = message.replace("\\n", "\n")
    splitedMsg = message.split("$")
    partitions = read_csv(StringIO(splitedMsg[0][2:]))
    weighting = splitedMsg[1]
    test = read_csv(StringIO(splitedMsg[2][:-1]))
    return partitions, weighting, test

#ENTRENAMIENTO Y CLASIFICACION

def classify(partition, weighting, test):
    #obtenemos distancia entre test y partition
    partitionList = partition.drop('classes', axis=1).values.tolist() 
    testList = test.values.tolist() 
    if (weighting == "piw"):
        inverseDistance = [0 for _ in range(len(testList))]
        for i in range(len(testList)):
            inverseDistance[i] = 1 / partf.end([testList[i]], partitionList, isR=1)
    else:
        inverseDistance = 1 / partf.end(testList, partitionList, isR=1)
    #obtenemos belief values
    if (USEDCLASSIFIER == "knn"):
        classifierOutput = knn(partition, test)
    elif(USEDCLASSIFIER == "rf"):
        classifierOutput = rf(partition, test)
    elif(USEDCLASSIFIER == "xgb"):
        classifierOutput = xgb(partition, test)
    else:
        print("UNKNOWN CLASSIFIER")
        exit(0)
    #pesamos los belief values
    print("weighting strategy:", weighting)
    if (weighting == "piw"):
        for i in range(len(classifierOutput)):
            for j in range(len(classifierOutput[i])):
                classifierOutput[i][j] = classifierOutput[i][j] * inverseDistance[i]
    else:
        for i in range(len(classifierOutput)):
            for j in range(len(classifierOutput[i])):
                classifierOutput[i][j] = classifierOutput[i][j] * inverseDistance
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
    partition, weighting, test = extractData(str(msg.payload))
    classifiedData = classify(partition, weighting, test)
    print("pubish weighed belief values:\n", classifiedData)
    client.publish("results/" + CLASSIFIERID, classifiedData + "$" + USEDCLASSIFIER, 2)

client = mqtt.Client("clas_client_" + CLASSIFIERID, True)
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect(BROKER_IP, 1883)

client.loop_forever()