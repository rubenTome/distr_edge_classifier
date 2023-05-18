import time
iniTime = time.time()
import paho.mqtt.client as mqtt
import partitionfunctions_python as partf
import fine_analisis_python as finean
import sys

#cliente para crear y publicar las particiones, y recibir resultados
#ARRANCAR DESPUES DE LOS CLASIFICADORES 

#PARAMETROS 

#number of partitions
Pset = sys.argv[1].replace(" ", "").strip("][").split(",")
for i in range(len(Pset)):
    Pset[i] = int(Pset[i])

#array de weighed belief values
wbelief = {i:[] for i in Pset}
is_balanced = True

clasTime = {i:0 for i in Pset}

#size of the total dataset (subsampple)
NSET = int(sys.argv[2])
#size of the train set, thse size of the test set will be NSET - NTRAIN
NTRAIN = int(sys.argv[3])

#posibles valores: "pnw", "piw"
weighingStrategy = sys.argv[4]

dataset = sys.argv[5]

BROKER_IP = "192.168.1.140"

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
    resArr = [[] for _ in range(NSET - NTRAIN)]
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
    ds = partf.load_dataset(dataset, NSET, NSET - NTRAIN)
    global testClasses
    testClasses = ds["testclasses"]
    #creamos las particiones segun parametro is_balanced
    if is_balanced:
        partitionFun = partf.create_random_partition
    else:
        partitionFun = partf.create_perturbated_partition
    partitions = [[] for _ in range(len(Pset))]
    test = ds["testset"]
    for p in range(len(Pset)):
        #creamos particiones para cada nodo
        partitions[p] = partitionFun(ds["trainset"], ds["trainclasses"], Pset[p])
    return (partitions, test)

def distClass(usedClassifier, clasTime, secondTime):
    splitedName = dataset.split("/")
    dsName = splitedName[len(splitedName) - 1].split(".")[0]
    file = open("rdos_" + str(NSET) + "_" + usedClassifier + "_" + weighingStrategy + "_" + dsName + "_distr.txt", "w")
    file.write("From dataset " + dataset + "\n")
    classArr = {i:[] for i in Pset}
    tempArr = []
    for i in (Pset):
        for j in range(len(wbelief[i][0])):
            for k in range(i):
                tempArr.append(wbelief[i][k][j])
            if (len(tempArr[0]) != 0):
                classArr[i].append(finean.sum_rule(tempArr) + 1)
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
    client.publish("exit", 1)

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
    client.subscribe("results/#")
    client.subscribe("exit")
    #comprobamos que la weighting strategy sea correcta
    if (weighingStrategy != "pnw" and weighingStrategy != "piw"):
        print("INVALID WEIGHING STRATEGY")
        client.publish("exit", 1)
    partAndTest = create_partitions()
    for j in range(len(Pset)):
        for k in range(Pset[j]):
            message = dataframeToStr(partAndTest[0][j][k]) + "$" + weighingStrategy + "$" + dataframeToStr(partAndTest[1])
            client.publish("partition/" + str(Pset[j]) + "." + str(k), message)
            print("published partition " + str(Pset[j]) + "." + str(k))

def on_message(client, userdata, msg):
    if (msg.topic == "exit"):
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

client = mqtt.Client("partitions_client")
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_IP, 1883)

client.loop_forever()