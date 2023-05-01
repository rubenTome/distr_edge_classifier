import paho.mqtt.client as mqtt
import partitionfunctions_python as partf
import pandas as pd
import sys

#cliente para crear y publicar las particiones, y recibir resultados
#ARRANCAR DESPUES DE LOS CLASIFICADORES 

#PARAMETROS 

#size of the total dataset (subsampple)
NSET = 1000
#size of the train set, thse size of the test set will be NSET - NTRAIN
NTRAIN = 500

#number of partitions
Pset = sys.argv[1].replace(" ", "").strip("][").split(",")
for i in range(len(Pset)):
    Pset[i] = int(Pset[i])

#array de weighed belief values
wbelief = {i:[] for i in Pset}
print(wbelief)
is_balanced = True

datasets = ["../scenariosimul/scenariosimulC2D2G3STDEV0.15.csv"]

#some datasets are split into train and test, because of concept drift
testdatasets= [""]

BROKER_IP = "192.168.1.140"

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

def create_partitions():
    for d in range(len(datasets)):
        ds = partf.load_dataset(datasets[d], NSET, NSET - NTRAIN)
        #creamos las particiones segun parametro is_balanced
        if is_balanced:
            partitionFun = partf.create_random_partition
        else:
            partitionFun = partf.create_perturbated_partition
        partitions = [[] for _ in range(len(Pset))]
        test = ds["testset"]
        distances = [[] for _ in range(len(Pset))]
        for p in range(len(Pset)):
            #creamos particiones para cada nodo
            partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p])
        for i in range(len(Pset)):
            for j in range(Pset[i]):
                #medimos distancia entre cada particion y el conjunto global de datos
                distances[i].append(partf.end(partitions[i][j].drop('classes', axis=1).values.tolist(),
                                              ds["trainset"].values.tolist()))
    return (partitions, distances, test)

#MQTT
def on_connect(client, userdata, flags, rc):
    print("Connected partitions client with result code " + str(rc))
    client.subscribe("results/#")
    partAndDist = create_partitions()
    for i in range(len(datasets)):
        for j in range(len(Pset)):
            for k in range(Pset[j]):
                name = datasets[i].split("/")
                name = name[len(name) - 1]
                message = dataframeToStr(partAndDist[0][j][k]) + "$" + str(partAndDist[1][j][k]) + "$" + dataframeToStr(partAndDist[2]) + "$" + name
                client.publish("partition/" + str(Pset[j]) + "." + str(k), message)
                print("published partition " + str(Pset[j]) + "." + str(k))

def on_message(client, userdata, msg):
    nPset = int(msg.topic.split("/")[1].split(".")[0])
    wbelief[nPset].append(str(msg.payload).replace("\\n", "\n")[2:-1])
    print("from topic " + msg.topic + ": received weighed belief values")
    allReceived = 1
    for i in Pset:
        if (len(wbelief[i]) != i):
            allReceived = 0
            break
    if (allReceived):
        print("all received")

client = mqtt.Client("partitions_client")
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, 1883)

client.loop_forever()