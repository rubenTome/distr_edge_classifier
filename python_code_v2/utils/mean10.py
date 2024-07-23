import sys

file = sys.argv[1]
outFile = open("mean10.txt", "w")

#models and weightings from classifierLoop.py
models = ["knn",
          "rf",
          "svm",
          "xgb"]
weightings = ["now",
              "pnw",
              "piwm"]
mw = [i + " " + j for i in models for j in weightings][::-1]

nNodes = ""
dataset = ""
acc = []
prec = []
rec = []
t = []
classifier = ""

def writeMean(acc, prec, rec, t, classifier):
    outFile.write(nNodes)
    outFile.write(dataset)
    outFile.write(classifier + "\n")
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
            #line with the number of nodes
            if n == 0:
                nNodes = lines[i]
        if i % 7 == 1 and n == 0:
            #lines with dataset and classifiers parameters
            dataset = lines[i]
            classifier = mw.pop()
        #lines with accuracy precision and recall
        if i % 7 == 2:
            acc.append(float(lines[i].split(":")[1]))
        if i % 7 == 3:
            prec.append(float(lines[i].split(":")[1]))
        if i % 7 == 4:
            rec.append(float(lines[i].split(":")[1]))
        if i % 7 == 5:
            t.append(float(lines[i].split(":")[1]))
        if i % 7 == 6:
            #skip last line
            i += 1
        #write results
        if n == 10:
            if mw == []:
                mw = [j + " " + k for j in models for k in weightings][::-1]
            writeMean(acc, prec, rec, t, classifier)
            acc = []
            prec = []
            rec = []
            t = []
            n = 0
    writeMean(acc, prec, rec, t, classifier)
