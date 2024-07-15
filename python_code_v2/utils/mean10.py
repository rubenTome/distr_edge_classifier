import sys

file = sys.argv[1]
outFile = open("mean10.txt", "w")

nNodes = ""
dataset = ""
acc = []
prec = []
rec = []
t = []

def writeMean(acc, prec, rec, t):
    outFile.write(nNodes)
    outFile.write(dataset)
    outFile.write("\tmean acc:" + str(sum(acc)/len(acc)) + "\n")
    outFile.write("\tmean prec:" + str(sum(prec)/len(prec)) + "\n")
    outFile.write("\tmean rec:" + str(sum(rec)/len(rec)) + "\n")
    outFile.write("\tmean time:" + str(sum(t)/len(t)) + "\n")
    outFile.write("----------------\n")

with open(file, 'r') as f:
    lines = f.readlines()
    n = -1
    for i in range(len(lines)):
        if i % 7 == 0:
            n += 1
            if n == 0:
                nNodes = lines[i]
        if i % 7 == 1 and n == 0:
            dataset = lines[i]
        if i % 7 == 2:
            acc.append(float(lines[i].split(":")[1]))
        if i % 7 == 3:
            prec.append(float(lines[i].split(":")[1]))
        if i % 7 == 4:
            rec.append(float(lines[i].split(":")[1]))
        if i % 7 == 5:
            t.append(float(lines[i].split(":")[1]))
        if i % 7 == 6:
            i += 1
        if n == 10:
            writeMean(acc, prec, rec, t)
            acc = []
            prec = []
            rec = []
            t = []
            n = 0
    writeMean(acc, prec, rec, t)
