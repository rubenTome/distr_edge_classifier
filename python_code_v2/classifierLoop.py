import os
import sys

nodeTopic = sys.argv[1]
model = sys.argv[2]
brokerIp = sys.argv[3]
weighting = sys.argv[4]

while(True):
    statement = "python3 classifierNode.py " + nodeTopic + " " + model + " " + brokerIp + " " + weighting
    print("loop:", statement)
    os.system(statement)