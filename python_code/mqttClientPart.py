import paho.mqtt.client as mqtt
import partitionfunctions_python as partf
import numpy as np
import json

#cliente para crear y publicar las particiones, y recibir resultados
#ARRANCAR DESPUES DE LOS CLASIFICADORES 

#PARAMETROS 

#size of the total dataset (subsampple)
NSET = 1000
#size of the train set, thse size of the test set will be NSET - NTRAIN
NTRAIN = 500

#number of partitions
Pset = [1]

is_balanced = True

datasets = ["../scenariosimul/scenariosimulC2D2G3STDEV0.15.csv", 
            "../scenariosimul/scenariosimulC8D3G3STDEV0.05.csv"]

#some datasets are split into train and test, because of concept drift
testdatasets= [""]

BROKER_IP = "192.168.1.143"

#CREACION PARTICIONES

def dataframeToStr(df):
    #nombres de las columnas de cada particion
    string = ",".join(df.columns) + "\n"
    for k in range(len(df.values)):
        #valores de las filas de cada particion
        string = string + ",".join([str(l) for l in df.values[k]]) + "\n"
    return string

def datasetToStr(ds):
    keys = ["trainset", "trainclasses", "testset", "testclasses"]
    string = ""
    for i in range(len(ds)):
        if i == 1 or i == 3:
            string = string + keys[i] + ": " + dataframeToStr(ds[keys[i]].to_frame())
        else:
            string = string + keys[i] + ": " + dataframeToStr(ds[keys[i]])
        if i < len(ds) - 1:
            string = string + ";"
    return string

def partition():
    for d in range(len(datasets)):
        ds = partf.load_dataset(datasets[d], NSET, NSET - NTRAIN)
        #creamos las particiones segun parametro is_balanced
        if is_balanced:
            partitionFun = partf.create_random_partition
        else:
            partitionFun = partf.create_perturbated_partition
        partitions = [[] for _ in range(len(Pset))]
        for p in range(len(Pset)):
                partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p])
        for i in range(len(Pset)):
            for j in range(Pset[i]):
                dfStr = dataframeToStr(partitions[i][j])
                #pasamos el dataset original a todos los clasificadores
                #TODO revisar parametros innecesarios
                dfStr = dfStr + "$" + str(Pset) + "$" + str(datasets[d]) + "$" + str(d - 1) + "$" + datasetToStr(ds)
                #enviamos particiones
                client.publish("partition/" + str(i) + "." + str(j), dfStr)
                print("\npublished partition" + str(i) + "." + str(j))
                dfStr = ""

#MQTT
def on_connect(client, userdata, flags, rc):
    print("Connected partitions client with result code " + str(rc))
    client.subscribe("partition/results/#")
    partition()

def on_message(client, userdata, msg):
    print("\nfrom topic " + msg.topic + ":\n" + str(msg.payload).replace("\\n", "\n"))

client = mqtt.Client("partitions_client")
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, 1883)

client.loop_forever()