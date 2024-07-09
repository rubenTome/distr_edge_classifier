import os
import sys

nodeTopic = sys.argv[1]
model = sys.argv[2]
brokerIp = sys.argv[3]
weighting = sys.argv[4]
N = sys.argv[5]

for _ in range(int(N)):
    statement = "python3 classifierNode.py " + nodeTopic + " " + model + " " + brokerIp + " " + weighting
    print("loop:", statement)
    os.system(statement)
