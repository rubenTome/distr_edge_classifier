import paho.mqtt.client as mqtt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score
import partitionfunctions_python as partf
#import fine_analisis_python as fan
import numpy as np
import csv
from prettytable import PrettyTable

#los clientes se subscriben a su particion y publican los resultados
#ARRANCAR PRIMERO LOS CLASIFICADORES

# #CLASIFICADORES

# def knn(partition):#partition es un pandas.DataFrame
#     partition = partition.to_numpy()
#     nVars = np.shape(partition)[1] - 1
#     trainset = partition[:, np.arange(nVars)]
#     trainclasses = partition[:,[nVars]].flatten()
#     clf = KNeighborsClassifier(n_neighbors = 2)
#     clf.fit(trainset, trainclasses)
#     score = clf.score(ds["testset"].to_numpy(), ds["testclasses"].to_numpy().flatten())
#     if(len(np.unique(trainclasses)) == 2):
#         av = "binary"
#     else:
#         av = "weighted"
#     precision = precision_score(ds["testclasses"].to_numpy().flatten(), 
#                                 clf.predict(ds["testset"].to_numpy()), average=av)
#     recall = recall_score(ds["testclasses"].to_numpy().flatten(), 
#                           clf.predict(ds["testset"].to_numpy()), average=av)

#     return [score, precision, recall]

# def rf(partition):
#     partition = partition.to_numpy()
#     nVars = np.shape(partition)[1] - 1
#     trainset = partition[:, np.arange(nVars)]
#     trainclasses = partition[:,[nVars]].flatten()
#     rfc = RandomForestClassifier()
#     rfc.fit(trainset, trainclasses)
#     scores = rfc.score(ds["testset"].to_numpy(), ds["testclasses"].to_numpy().flatten())
#     if(len(np.unique(trainclasses)) == 2):
#         av = "binary"
#     else:
#         av = "weighted"
#     precision = precision_score(ds["testclasses"].to_numpy().flatten(), 
#                                 rfc.predict(ds["testset"].to_numpy()), average=av)
#     recall = recall_score(ds["testclasses"].to_numpy().flatten(), 
#                           rfc.predict(ds["testset"].to_numpy()), average=av)
#     return [scores, precision, recall]

# def xgb(partition):
#     partition = partition.to_numpy()
#     nVars = np.shape(partition)[1] - 1
#     trainset = partition[:, np.arange(nVars)]
#     trainclasses = partition[:,[nVars]].flatten()
#     gbc = GradientBoostingClassifier()
#     gbc.fit(trainset, trainclasses)
#     scores = gbc.score(ds["testset"].to_numpy(), ds["testclasses"].to_numpy().flatten())
#     if(len(np.unique(trainclasses)) == 2):
#         av = "binary"
#     else:
#         av = "weighted"
#     precision = precision_score(ds["testclasses"].to_numpy().flatten(), 
#                                 gbc.predict(ds["testset"].to_numpy()), average=av)
#     recall = recall_score(ds["testclasses"].to_numpy().flatten(), 
#                           gbc.predict(ds["testset"].to_numpy()), average=av)
#     return [scores, precision, recall]


# #PARAMETROS 

# totalresults = None

# #generar tablas con los resultados (+legible)
# generateTables = True

# classifiers = [knn, rf, xgb]

# #names for printing them
# namesclassifiers = ["KNN", "RF", "XGB"] 

BROKER_IP = "192.168.1.138"

# #CREACION DE CLASIFICADORES
# classifAcc = {}
# for ps in range(len(Pset)):
#     for c in range(len(classifiers)):
#         for pa in range(len(partitions[ps])):
#             classifAcc[namesclassifiers[c] + "_" + str(Pset[ps]) + "_" + str(pa + 1)] = (
#             classifiers[c](partitions[ps][pa]))

# #guardamos en un csv la informacion de cada clasificador en classifAcc
# splitedPath = datasets[d].split("/")
# name = splitedPath[len(splitedPath) - 1]
# file = open("rdos_python/rdos_" + name, "w")
# if generateTables:
#     fileTable = open("rdos_python/tabla_rdos_" + name.split(".")[0] + ".txt", "w")
#     headers = PrettyTable(["classifier", "npartitions", "partition", "accurancy", "precision", "recall", "energy"])
#     headers.align = "l"
# writer = csv.writer(file)
# writer.writerow(["classifier", "npartitions", "partition", "accurancy", "precision", "recall", "energy"])
# for t in range(len(classifAcc)):
#     label = list(classifAcc)[t]
#     classifier = classifAcc[label]
#     splitedName = label.split("_")
#     actualData = partitions[Pset.index(int(splitedName[1]))][int(splitedName[2]) - 1].to_numpy()
#     actualData = actualData[:, np.arange(np.shape(actualData)[1] - 1)]
#     energyDist = partf.end(actualData, ds["trainset"].values.tolist(), True)
#     if generateTables:
#         headers.add_row([splitedName[0], splitedName[1], splitedName[2], 
#                         round(classifier[0], NDECIMALS), round(classifier[1], NDECIMALS), 
#                         round(classifier[2], NDECIMALS), round(energyDist, NDECIMALS)])
#     writer.writerow([splitedName[0], splitedName[1], splitedName[2], 
#                     round(classifier[0], NDECIMALS), round(classifier[1], NDECIMALS), 
#                     round(classifier[2], NDECIMALS), round(energyDist, NDECIMALS)])
# if generateTables:
#     fileTable.write(str(headers))

#MQTT
#se llama al conectarse al broker
def on_connect(client, userdata, flags, rc):
    print("Connected classification client with result code " + str(rc))
    client.subscribe("partition/1") #nos subscribimos a este tema

#se llama al obtener un mensaje del broker
def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload)) #imprimimos la respuesta
    client.publish("partition/results/1", "results_clas_client_1")#publicamos los resultados
    print("Published results: results_clas_client_1")

client = mqtt.Client("clas_client_1")
client.on_connect = on_connect
client.on_message = on_message

#conectamos con el broker
client.connect(BROKER_IP, 1883)#debe ser el nombre o ip

client.loop_forever()