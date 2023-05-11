import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from io import StringIO
import sys
import os
import signal

#los clientes se subscriben a su particion y publican los resultados
#ARRANCAR PRIMERO LOS CLASIFICADORES

#CLASIFICADORES

def knn(partition, test):#partition es un pandas.DataFrame
    partition = partition.to_numpy()
    nVars = np.shape(partition)[1] - 1
    trainset = partition[:, np.arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    clf = KNeighborsClassifier(n_neighbors = 2)
    clf.fit(trainset, trainclasses)
    testClass = clf.predict_proba(test[:].values)
    return testClass

def rf(partition, test):
    partition = partition.to_numpy()
    nVars = np.shape(partition)[1] - 1
    trainset = partition[:, np.arange(nVars)]
    trainclasses = partition[:,[nVars]].flatten()
    rfc = RandomForestClassifier()
    rfc.fit(trainset, trainclasses)
    testClass = rfc.predict_proba(test[:].values)
    return testClass

def xgb(partition, test):
    partition = partition.to_numpy()
    nVars = np.shape(partition)[1] - 1
    trainset = partition[:, np.arange(nVars)]
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
BROKER_IP = "192.168.1.143"

CLASSIFIERID = sys.argv[1]
USEDCLASSIFIER = sys.argv[2]

def extractData(message):
    message = message.replace("\\n", "\n")
    splitedMsg = message.split("$")
    partitions = pd.read_csv(StringIO(splitedMsg[0][2:]))
    inverseDistance = float(splitedMsg[1])
    test = pd.read_csv(StringIO(splitedMsg[2][:-1]))
    return partitions, inverseDistance, test

#ENTRENAMIENTO Y CLASIFICACION

def classify(partition, inverseDistance, test):
    #TEMPORALMENTE SOLO KNN

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
    for i in range(len(classifierOutput)):
        for j in range(len(classifierOutput[i])):
            classifierOutput[i][j] = classifierOutput[i][j] * inverseDistance
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
    #nos subscribimos a este tema
    client.subscribe("partition/" + CLASSIFIERID)
    client.subscribe("exit")
    print("\nSubscribed to partition/" + CLASSIFIERID)

#se llama al obtener un mensaje del broker
def on_message(client, userdata, msg):
    if (msg.topic == "exit"):
        os.kill(os.getppid(), signal.SIGHUP)
    partition, inverseDistance, test = extractData(str(msg.payload))
    classifiedData = classify(partition, inverseDistance, test)
    print("pubish weighed belief values:\n", classifiedData)
    client.publish("results/" + CLASSIFIERID, classifiedData + "$" + USEDCLASSIFIER)

client = mqtt.Client("clas_client_" + CLASSIFIERID)
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect(BROKER_IP, 1883)

client.loop_forever()