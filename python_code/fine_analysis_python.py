import numpy as np

# Decision rules

#mejor regla
def sum_rule(beliefs):
    return np.argmax(np.sum(beliefs, axis=0))

def multi_precision(preds, true_classes):
    tpl = 0
    fpl = 0
    precis = []
    uniqueList, ind = np.unique(true_classes, return_index=True)
    uniqueList = uniqueList[np.argsort(ind)]
    for l in uniqueList:#unique de numpy ordena, en R no
        tpl = tpl + np.sum((preds == l) & (true_classes == l))
        fpl = fpl + np.sum((preds == l) & (true_classes != l))
        if tpl + fpl < 1:
            fpl = 1
        precis.append(tpl / (tpl + fpl))
    return np.mean(precis)

def multi_recall(preds, true_classes):
    tpl = 0
    fnl = 0
    recalls = []
    uniqueList, ind = np.unique(true_classes, return_index=True)
    uniqueList = uniqueList[np.argsort(ind)]
    for l in uniqueList:
        tpl = tpl + np.sum((preds == l) & (true_classes == l))
        fnl = fnl + np.sum((preds != l) & (true_classes == l))
        recalls.append(tpl / (tpl + fnl))
    return np.mean(recalls)

def accu(preds, true_classes):
    return np.mean((np.array(preds) == np.array(true_classes)))