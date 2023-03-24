#Decision rules

#beliefs is a matrix with one column per partition, one row per class
max.rule = function(beliefs) {
  which.max(apply(beliefs, 1, max))
}

sum.rule = function(beliefs) {
  which.max(rowSums(beliefs))
}

product.rule = function(beliefs) {
  which.max(apply(beliefs, 1, prod))
}

min.rule = function(beliefs) {
  which.max(apply(beliefs, 1, min))
}

median.rule = function(beliefs) {
  which.max(apply(beliefs, 1, mean))
}

majority.rule = function(beliefs, weights=1) {
  beliefs = apply(beliefs, 2, function(x)
    x == max(x))
  beliefs = beliefs * weights
  which.max(rowSums(beliefs))
}

reverse.majority.rule = function(beliefs, reverseweights) {
  reversebeliefs = list(NULL)
  for (i in 1:length(beliefs)) {
    reversebeliefs[[i]] = beliefs[[i]]
    for (j in 1:ncol(beliefs[[i]])) {
      reversebeliefs[[i]][, j] =
        (reversebeliefs[[i]][, j] == max(reversebeliefs[[i]][, j]))* reverseweights[[j]]$weights[i]
    }
  }
  sapply(reversebeliefs, sum.rule)
}



multi_precision = function(preds, true_classes) {
  tpl = 0
  fpl= 0
  precis = NULL
  for (l in unique(true_classes)) {
    tpl = tpl + sum((preds == l) & (true_classes == l))
    fpl = fpl + sum((preds == l) & (true_classes != l))
    if( tpl + fpl < 1) fpl=1
    precis = c(precis, tpl / (tpl + fpl))
  }

  mean(precis)
}

multi_recall = function(preds, true_classes) {
  tpl = 0
  fnl= 0
  recalls = NULL
  for (l in unique(true_classes)) {
    tpl = tpl + sum((preds == l) & (true_classes == l))
    fnl = fnl + sum((preds != l) & (true_classes == l))
    recalls = c(recalls,   tpl / (tpl + fnl))
  }
  mean(recalls)
}

accu = function(preds, true_classes) {
  mean(preds==true_classes)
}

##################################################################
################### PROCESSING FILE ##############################
##################################################################

prefix_filename <- "unbal" #"unbal"

load(paste("5class_", prefix_filename, "_fine_grained.RData",
          sep=""))

namesclassifiers = c("forest", "svm", "xgboost", "multinom", "lda")
datasets = c("spambase", "simulC2",
             "simulC8", "connect4",
             "kdd",
             "covtype", "HIGGS")

##5 list inside, with the the classes prob for each classifier
#fine_results_list[[1]]

#the distance of each partition to the test set
#fine_results_list[[2]]

#the weights per partition per observation in the test (P list with $weights and $val)
#fine_results_list[[3]]

#weihts from observation to test, not used
#fine_results_list[[4]]

#the true classes
#fine_results_list[[5]]

#the number of partitions
#fine_results_list[[6]]

#the dataset index
#fine_results_list[[7]]

stats_by_sample <- NULL

while (length(fine_results_list) > 6) {
  print(length(fine_results_list))
  dataset = NULL
  dataset$testset = matrix(0, nrow=nrow((fine_results_list[[1]][[1]][[1]])), ncol=2)
  dataset$testclasses = fine_results_list[[5]]
  distances = fine_results_list[[2]]
  wreverse = fine_results_list[[3]]
  resultsbyclassifier <- fine_results_list[[1]]
  classifiers = 1:5
  P = fine_results_list[[6]]

  stats_by_classifier <- NULL
  for (ci in 1:length(classifiers)) {
    #combine classifier probs, distances to node and weights

    #extract the beliefs
    beliefs = NULL
    for (i in 1:nrow(dataset$testset)) {
      beliefs[[i]] = sapply(1:P, function (p) {
        resultsbyclassifier[[ci]][[p]][i, ]
      })
    }

    #baselines

    #decision modulated by distance
    distbeliefs = lapply(beliefs, function(bel) {
      for (j in 1:ncol(bel)) {
        bel[, j] = (bel[, j]) * ((1 / distances[j]) /  sum(1 / distances))
      }
      bel
    })

    #decision modulated by reverse weight
    reversebeliefs = list(NULL)
    for (i in 1:length(beliefs)) {
      reversebeliefs[[i]] = beliefs[[i]]
      for (j in 1:ncol(distbeliefs[[i]])) {
        reversebeliefs[[i]][, j] = reversebeliefs[[i]][, j] * wreverse[[j]]$weights[i]
      }
    }

    pred.sum.unw = sapply(beliefs, sum.rule)
    pred.prod.unw = sapply(beliefs, product.rule)
    pred.maj.unw = sapply(beliefs, majority.rule)
    pred.min.unw = sapply(beliefs, min.rule)
    pred.max.unw = sapply(beliefs, max.rule)

    pred.sum.dist = sapply(distbeliefs, sum.rule)
    pred.prod.dist = sapply(distbeliefs, product.rule)
    pred.maj.dist = sapply(distbeliefs, majority.rule,
                           weights = (1 / distances) /  sum(1 / distances) )
    pred.min.dist = sapply(distbeliefs, min.rule)
    pred.max.dist = sapply(distbeliefs, max.rule)

    pred.sum.rev = sapply(reversebeliefs, sum.rule)
    pred.prod.rev = sapply(reversebeliefs, product.rule)
    pred.maj.rev = reverse.majority.rule(beliefs, wreverse)
    pred.min.rev = sapply(reversebeliefs, min.rule)
    pred.max.rev = sapply(reversebeliefs, max.rule)

    precis_line <- c(multi_precision(pred.sum.unw, dataset$testclasses),
                      multi_precision(pred.prod.unw, dataset$testclasses),
                      multi_precision(pred.maj.unw, dataset$testclasses),
                      multi_precision(pred.min.unw, dataset$testclasses),
                      multi_precision(pred.max.unw, dataset$testclasses),
                      multi_precision(pred.sum.dist, dataset$testclasses),
                      multi_precision(pred.prod.dist, dataset$testclasses),
                      multi_precision(pred.maj.dist, dataset$testclasses),
                      multi_precision(pred.min.dist, dataset$testclasses),
                      multi_precision(pred.max.dist, dataset$testclasses),
                      multi_precision(pred.sum.rev, dataset$testclasses),
                      multi_precision(pred.prod.rev, dataset$testclasses),
                      multi_precision(pred.maj.rev, dataset$testclasses),
                      multi_precision(pred.min.rev, dataset$testclasses),
                      multi_precision(pred.max.rev, dataset$testclasses))

    recall_line <- c(multi_recall(pred.sum.unw, dataset$testclasses),
                     multi_recall(pred.prod.unw, dataset$testclasses),
                     multi_recall(pred.maj.unw, dataset$testclasses),
                     multi_recall(pred.min.unw, dataset$testclasses),
                     multi_recall(pred.max.unw, dataset$testclasses),
                     multi_recall(pred.sum.dist, dataset$testclasses),
                     multi_recall(pred.prod.dist, dataset$testclasses),
                     multi_recall(pred.maj.dist, dataset$testclasses),
                     multi_recall(pred.min.dist, dataset$testclasses),
                     multi_recall(pred.max.dist, dataset$testclasses),
                     multi_recall(pred.sum.rev, dataset$testclasses),
                     multi_recall(pred.prod.rev, dataset$testclasses),
                     multi_recall(pred.maj.rev, dataset$testclasses),
                     multi_recall(pred.min.rev, dataset$testclasses),
                     multi_recall(pred.max.rev, dataset$testclasses))

    accu_line <- c(accu(pred.sum.unw, dataset$testclasses),
                   accu(pred.prod.unw, dataset$testclasses),
                   accu(pred.maj.unw, dataset$testclasses),
                   accu(pred.min.unw, dataset$testclasses),
                   accu(pred.max.unw, dataset$testclasses),
                   accu(pred.sum.dist, dataset$testclasses),
                   accu(pred.prod.dist, dataset$testclasses),
                   accu(pred.maj.dist, dataset$testclasses),
                   accu(pred.min.dist, dataset$testclasses),
                   accu(pred.max.dist, dataset$testclasses),
                   accu(pred.sum.rev, dataset$testclasses),
                   accu(pred.prod.rev, dataset$testclasses),
                   accu(pred.maj.rev, dataset$testclasses),
                   accu(pred.min.rev, dataset$testclasses),
                   accu(pred.max.rev, dataset$testclasses))

    stopifnot( sum(is.nan(precis_line))==0)
    stats_row <- rbind(c("precision", precis_line),
                       c("recall", recall_line),
                       c("accuracy", accu_line))
    stats_row <- cbind(namesclassifiers[[ci]], stats_row)
    stats_by_classifier <- rbind(stats_by_classifier, stats_row)
  }

  stats_by_sample <- rbind(stats_by_sample,
                           cbind(datasets[fine_results_list[[7]]],
                                 fine_results_list[[6]], stats_by_classifier))
  colnames(stats_by_sample) <- c("DATASET", "P", "CLASSIFIER", "STAT",
                                 "BASE.SUM", "BASE.PROD", "BASE.MAJ", "BASE.MIN", "BASE.MAX",
                                 "DIST.SUM", "DIST.PROD", "DIST.MAJ", "DIST.MIN", "DIST.MAX",
                                 "REV.SUM", "REV.PROD", "REV.MAJ", "REV.MIN", "REV.MAX")

  fine_results_list = fine_results_list[-(1:7)]
}

write.table(stats_by_sample, file="stats_results.csv", col.names = TRUE, row.names = FALSE, sep=",")


a <- read.csv("stats_results.csv")

#calculate the means
stat_mean<-aggregate(a[-(1:4)], list(a$DATASET, a$P, a$CLASSIFIER, a$STAT ), mean)
colnames(stat_mean) <- colnames(a)
write.table(stat_mean, file="stats_means.csv", col.names = TRUE, row.names = FALSE, sep=",")



##global results
load("5class_global.RData")
Pset = c(2,4,7,11,15)
global_lines <- NULL
while (length(global_results_list) > 3) {
  res_by_classif <- global_results_list[[1]]
  dataset <- global_results_list[[2]]
  classifiers <- global_results_list[[3]]
  true_classes <- global_results_list[[4]]

  for (ci in 1:length(res_by_classif)) {
    preds <- res_by_classif[[ci]]
    preds <- apply(preds, 1, which.max)
    global_prec <- multi_precision(preds, true_classes)
    global_recall <- multi_recall(preds, true_classes)
    global_accu <- accu(preds, true_classes)
    stat_line <- c(global_prec, global_recall, global_accu)
    res_line <- NULL
    for (P in Pset) {
      res_line <- rbind(res_line, cbind(dataset, P, classifiers[[ci]],
                                        c("precision", "recall", "accuracy"),
                                        stat_line) )
    }
    colnames(res_line) <- c("DATASET", "P", "CLASSIFIER", "STAT", "GLOBAL")
    global_lines <- rbind(global_lines, res_line)
  }
  global_results_list <- global_results_list[-(1:4)]
}
colnames(global_lines) <- c("DATASET", "P", "CLASSIFIER", "STAT", "GLOBAL")
global_lines <- as.data.frame(global_lines)
write.table(global_lines, file="gltemp.csv", col.names = TRUE, row.names = FALSE, sep=",")
global_lines <- read.csv("gltemp.csv")
global_lines[,1] <- as.character(global_lines[,1])
global_lines[global_lines[,1] =="scenariosimul/scenariosimulC8D3G3STDEV0.05.csv" ,1] <- "simulC8"
global_lines[global_lines[,1] =="scenariosimul/scenariosimulC2D5G3STDEV0.05.csv" ,1] <- "simulC2"
global_lines[global_lines[,1] =="scenariosimul/HIGGS.csv" ,1] <- "HIGGS"
global_lines[global_lines[,1] =="scenariosimul/covtype.data" ,1] <- "covtype"
global_lines[global_lines[,1] =="scenariosimul/connect-4Train.csv" ,1] <- "connect4"
global_lines[global_lines[,1] =="scenariosimul/cleankdd_train.csv" ,1] <- "kdd"
global_lines[global_lines[,1] =="scenariosimul/spambase.data" ,1] <- "spambase"


global_mean<-aggregate(as.numeric(as.character(global_lines[,-(1:4)])),
                       list(global_lines$DATASET,
                            global_lines$P,
                            global_lines$CLASSIFIER,
                            global_lines$STAT ), mean)

global_mean[,1] <- as.character(global_mean[,1])
global_mean[global_mean[,1] =="scenariosimul/scenariosimulC8D3G3STDEV0.05.csv" ,1] <- "simulC8"
global_mean[global_mean[,1] =="scenariosimul/scenariosimulC2D5G3STDEV0.05.csv" ,1] <- "simulC2"
global_mean[global_mean[,1] =="scenariosimul/HIGGS.csv" ,1] <- "HIGGS"
global_mean[global_mean[,1] =="scenariosimul/covtype.data" ,1] <- "covtype"
global_mean[global_mean[,1] =="scenariosimul/connect-4Train.csv" ,1] <- "connect4"
global_mean[global_mean[,1] =="scenariosimul/cleankdd_train.csv" ,1] <- "kdd"
global_mean[global_mean[,1] =="scenariosimul/spambase.data" ,1] <- "spambase"
write.table(global_mean, file="global_means.csv", col.names = TRUE, row.names = FALSE, sep=",")


a <- read.csv("global_means.csv")
b <- read.csv("stats_means.csv")
mean(a[,1:4] == b[,1:4])

b <- cbind(b, a[,ncol(a)])
colnames(b)[ncol(b)] <- "GLOBAL"


write.table(b, file=paste("5c_", prefix_filename, "_mean_results.csv", sep=""),
            col.names = TRUE, row.names = FALSE, sep=",")
