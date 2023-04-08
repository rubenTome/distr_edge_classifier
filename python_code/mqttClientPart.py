import paho.mqtt.client as mqtt
import partitionfunctions_python as partf
import numpy as np

#cliente para crear y publicar las particiones, y recibir resultados
#ARRANCAR DESPUES DE LOS CLASIFICADORES 

#PARAMETROS 

#size of the total dataset (subsampple)
NSET = 1000
#size of the train set, thse size of the test set will be NSET - NTRAIN
NTRAIN = 500

#number of partitions
Pset = [1, 4]

is_balanced = True

datasets = ["../scenariosimul/scenariosimulC2D2G3STDEV0.15.csv", 
            "../scenariosimul/scenariosimulC8D3G3STDEV0.05.csv"]

#some datasets are split into train and test, because of concept drift
testdatasets= [""]

BROKER_IP = "192.168.1.138"

#MQTT
def on_connect(client, userdata, flags, rc):
    print("Connected partitions client with result code " + str(rc))
    client.subscribe("partition/results/#")
    print("Subscribed to partition/results/#")
    #CREACION PARTICIONES
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
    #TEMPORALMENTE NOS QUEDAMOS SOLO CON EL ULTIMO DATASET
    for i in range(len(Pset)):
        for j in range(Pset[i]):
            #nombres de las columnas de cada particion
            dfStr = ",".join(partitions[i][j].columns) + "\n"
            for k in range(len(partitions[i][j].values)):
                #valores de las filas de cada particion
                dfStr = dfStr + ",".join([str(l) for l in partitions[i][j].values[k]]) + "\n"
            #enviamos particiones
            client.publish("partition/" + str(i) + "." + str(j), dfStr)
            print("publicada particion" + str(i) + "." + str(j))
            dfStr = ""

def on_message(client, userdata, msg):
    print(msg.topic + " " + str(msg.payload))

client = mqtt.Client("partitions_client")
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, 1883)

client.loop_forever()