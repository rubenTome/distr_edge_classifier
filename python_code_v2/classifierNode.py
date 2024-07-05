import paho.mqtt.client as mqtt
import socket
import sys
import utils.metrics as metrics
import utils.data_loaders as data_loaders
import python_code_v2.utils.weighting as weighting
import utils.classifiers as classifiers
import pandas as pd
import numpy as np
import io

BROKER_IP = sys.argv[3]
PORT = 1883

def on_connect(client, userdata, flags, rc):
    client.unsubscribe("#")
    print("connected to mqtt broker with code", rc)
    client.subscribe("exit")
    print("subscribed to exit")
    #subscribe to the topic of node train subset
    client.subscribe(sys.argv[1])
    print("subscribed to", sys.argv[1])

def on_message(client, userdata, msg):
    #diconnect node
    if (msg.topic == "exit"):
        print("exiting...")
        classNode.unsubscribe("#")
        client.disconnect()
        exit(0)
    #receive data from central node, train and classify data
    if (msg.topic == sys.argv[1]):
        print(msg.topic + ": received train subset and test data")
        dataStr = str(msg.payload)
        #unpack data and convert string to pandas dataframes
        trainData, testData = dataStr.split("$")
        #remove first 3 characters and replace "\n" string with new line
        trainData = trainData[2:]
        trainData = trainData.replace("\\n", "\n")
        #remove last character and replace "\n" string with new line
        testData = testData[:-1]
        testData = testData.replace("\\n", "\n")
        trainData = pd.read_csv(io.StringIO(trainData))
        testData = pd.read_csv(io.StringIO(testData))
        #classify data, return pandas series with predicted labels
        print("training...")
        if(sys.argv[2] == "knn"):
            predicted = classifiers.knn(trainData, testData)
        elif(sys.argv[2] == "xgb"):
            predicted = classifiers.xgb(trainData, testData)
        elif(sys.argv[2] == "rf"):
            predicted = classifiers.rf(trainData, testData)
        elif(sys.argv[2] == "svm"):
            predicted = classifiers.svm(trainData, testData)
        else:
            print("unknown classifier (correct values: knn, xgb, rf, svm)")
            print("exiting...")
            classNode.publish("exit", 1, 2)
            exit(-1)
        #wheight predicted values
        print("wheighting...")
        trainDataList = trainData.drop('classes', axis=1).values
        testDataList = testData.drop('classes', axis=1).values
        if(sys.argv[4] == "pnw"):
            wPredicted = weighting.pnw(predicted, trainDataList, testDataList)
        elif(sys.argv[4] == "piw"):
            wPredicted = weighting.piw(predicted, trainDataList, testDataList)
        elif(sys.argv[4] == "random"):
            wPredicted = weighting.random(predicted)
        else:
            print("unknown wheighting strategy (correct values: piw, pnw, random)")
            print("exiting...")
            classNode.publish("exit", 1, 2)
            exit(-1)
        #send array of wheighted predicted classes to central node
        wPredicted = pd.DataFrame(wPredicted).to_csv()
        classNode.publish(sys.argv[1] + ".results", wPredicted, 2)
        print("published weighted belief values")


classNode = mqtt.Client("classNode", True)
classNode.on_connect = on_connect
classNode.on_message = on_message
classNode.connect(BROKER_IP, PORT)
classNode.loop_forever()