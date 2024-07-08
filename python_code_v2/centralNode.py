import time
timer = time.time()
import paho.mqtt.client as mqtt
import socket
import sys
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import utils.decisionRules as decisionRules
import pandas as pd
import numpy as np
import io

BROKER_IP = socket.gethostbyname(socket.gethostname())
PORT = 1883

#calculate accuracy, precision and recall
def computeMetrics(predicted, real):
    acuracy = metrics.accu(predicted, real)
    precision = metrics.multi_precision(predicted, real)
    recall = metrics.multi_recall(predicted, real)
    return acuracy, precision, recall

#select partition type (random, perturbated or selected)
def selectPartFun(str, nPartition, data):
    if str == "random":
        trainSets, testSet = data_loaders.create_random_partition(
            data, nPartition, float(sys.argv[3]), float(sys.argv[4]))
        return trainSets, testSet
    elif str == "perturbated":
        trainSets, testSet = data_loaders.create_perturbated_partition(
            data, nPartition, float(sys.argv[3]), float(sys.argv[4]))
        return trainSets, testSet
    elif str == "selected":
        #needed an extra argument to select classes distribution of the nodes
        trainSets, testSet = data_loaders.create_selected_partition(
            data, nPartition, sys.argv[7], float(sys.argv[3]), float(sys.argv[4]))
        return trainSets, testSet
    else:
        print("unknown partition type (correct values: random, perturbated, selected)")
        print("exiting...")
        exit(-1)

#create node topics given the number of classifier nodes
def createNodeTopics(nPartition):
    nodeTopics = []
    for i in range(int(nPartition)):
        nodeTopics.append(nPartition + "." + str(i))
    return nodeTopics

#dataframe to store results from each node
global nodeResults 
nodeResults = []
#number of train subsets, one per classifier node
nPartition = sys.argv[1]
#topics assigned to each node
nodeTopics = createNodeTopics(nPartition)
#results topics: X.X.results
nodeTopicsRes = [i + ".results" for i in nodeTopics]
#load and divide data
data = data_loaders.load_dataset(sys.argv[6], int(sys.argv[2]))
trainSets, testSet = selectPartFun(sys.argv[5], int(nPartition), data)

def on_connect(client, userdata, flags, rc):
    print("connected to mqtt broker with code", rc)
    for i in range(int(nPartition)):
        topic = str(nPartition) + "." + str(i)
        #subscribe to each node results topic
        client.subscribe(topic + ".results")
        print("subscribed to", topic + ".results")
        #send data to each node
        client.publish(topic, trainSets[i].to_csv(index=False) + "$" + testSet.to_csv(index=False), 2)
        print("published partition to node", topic)

def on_message(client, userdata, msg):
    #receive data from node and store it
    if msg.topic in nodeTopicsRes:
        nodeTopicsRes.remove(msg.topic)
        print("received classified data:", msg.topic)
        client.publish(msg.topic.replace("results", "") + "exit", 1, 2)
        print("publishing exit message:", msg.topic.replace("results", "") + "exit")
        #combine csv strings in a single dataframe
        receivedStr = str(msg.payload)
        #remove first 3 characters and replace "\n" string with new line
        receivedStr = receivedStr[3:]
        receivedStr = receivedStr.replace("\\n", "\n")
        nodeResults.append(pd.read_csv(io.StringIO(receivedStr)).dropna())
        #compute metrics when all data is received
        if nodeTopicsRes == []:
            print("all data received, merging results...")
            #apply the sum rule to combine results
            mergedResults = decisionRules.sum_rule(nodeResults)
            print("computing metrics...")
            acc, prec, rec = computeMetrics(np.array(mergedResults), testSet.iloc[:,-1:].to_numpy().flatten())
            #print metrics for each partition size
            execTime = time.time() - timer
            print("for", nPartition, "partitions:")
            print("\taccuracy:", acc)
            print("\tprecision:", prec)
            print("\trecall:", rec)
            #print execution time
            print("execution time:", execTime)
            #disconnect this node
            client.disconnect()
            print("exiting...")
            exit(0)

centralNode = mqtt.Client("centralNode", True)
centralNode.on_connect = on_connect
centralNode.on_message = on_message
centralNode.connect(BROKER_IP, PORT)
centralNode.loop_forever()