import os
import sys
import time

nNodes = sys.argv[1]
nData = sys.argv[2]
train = sys.argv[3]
test = sys.argv[4]
partition = sys.argv[5]
dataset = sys.argv[6]
N =sys.argv[7]

for _ in range(N):
    statement = "python3 centralNode.py " + nNodes + " " + nData + " " + train + " " + test + " " + partition + " " + dataset
    print("loop:", statement)
    os.system(statement)
    time.sleep(1)