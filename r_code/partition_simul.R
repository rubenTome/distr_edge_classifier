

#####################################################
####   classifiers outputting degrees of belief  ####
#####################################################
#1º knn3, 2ºrf, xgb, mult 
svm.classifier.prob = function(trainset,
                               trainclasses,
                               testset,
                               testclasses) {
  SVM = e1071::svm(
    x = trainset,
    y = as.factor(trainclasses),
    scale = FALSE,
    probability = TRUE
  )
  prob = attr(predict(SVM, testset, probability = TRUE), "prob")
  prob = prob[, order(colnames(prob))]
  list(prob = prob)
}

forest.classifier.prob = function(trainset,
                                  trainclasses,
                                  testset,
                                  testclasses) {
  forestmodel = randomForest::randomForest(x = trainset, y = as.factor(trainclasses))
  list(prob = predict(forestmodel, testset, type = "prob"))
}


xgb.classifier.prob = function(trainset, trainclasses, testset, testclasses) {

  trainset = sapply(as.data.frame(trainset), as.numeric)
  testset = sapply(as.data.frame(testset), as.numeric)
  nclass = length(unique(c(trainclasses, testclasses)))
  trainclasses = trainclasses - 1
  testclasses = testclasses - 1

  dtrain <- xgboost::xgb.DMatrix(trainset, label = trainclasses )
  dtest <- xgboost::xgb.DMatrix(testset, label = testclasses )

  param <- list(max_depth = 6, eta = 0.3, #silent = 1, temporal, evita warning
                nthread = 2,
                min_child_weight = 1,
                num_class = nclass,
                objective = "multi:softprob")


   xgbmodel = xgboost::xgb.train(param, dtrain, nrounds=25)
   list(prob = predict(xgbmodel, newdata=dtest, outputmargin=FALSE, reshape=TRUE ))
}


multinom.classifier.prob = function(trainset, trainclasses, testset, testclasses) {

  response <- as.factor(trainclasses)
  model <- nnet::multinom(response ~ ., data=as.data.frame(trainset), trace=FALSE)
  preds <- predict(object=model, newdata = testset, type="probs")
  if ( length(unique(response)) == 2) {
    preds <- t(sapply(preds, function (p) c(1-p,p)))
  }
  list(prob = preds)
}

lda.classifier.prob = function(trainset, trainclasses, testset, testclasses) {

  response <- as.factor(trainclasses)
  trainset <- trainset + rnorm(length(trainset), 0, 0.001)
  model <- MASS::lda(response~., data=as.data.frame(trainset))
  testset <- testset + rnorm(length(testset), 0, 0.001)
  preds = predict(model, as.data.frame(testset))$posterior

  list(prob = preds)

}

######################################
#######  Decision rules  #############
######################################

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

reverse.majority.rule = function(originalbeliefs, reverseweights) {
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



#############################################################
######## PARAMETERS OF THE SIMULATION #######################
#############################################################


source("partitionfunctions.R")

totalresults = NULL


NREP = 5#15 #how many reps per experiment
nreplicates = 1  #leave this at one :)
NSET = 500#1000    #size of the total dataset (subsampple)
NTRAIN = 250#500   #size of the train set, thse size of the test set will be NSET - NTRAIN

#number of partitions
Pset = c(2,4,7,11,15)

is_balanced = TRUE

datasets = c("../scenariosimul/spambase.data", "../scenariosimul/scenariosimulC2D5G3STDEV0.05.csv",
             "../scenariosimul/scenariosimulC8D3G3STDEV0.05.csv", "../scenariosimul/connect-4Train.csv",
             "../scenariosimul/cleankdd_train.csv",
             "../scenariosimul/covtype.data", "../scenariosimul/HIGGS.csv")

#some datasets are split into train and test, because of concept drift
testdatasets= c("", "", "", "../scenariosimul/connect-4Test.csv",
                "../scenariosimul/cleankdd_test.csv",
                "", "")

#functions with the classifiers
classifiers = c(forest.classifier.prob, svm.classifier.prob,
                xgb.classifier.prob,
                multinom.classifier.prob, lda.classifier.prob)
#names for printing them
namesclassifiers = c("forest", "svm", "xgboost", "multinom", "lda")




####################################################
######## BEGINNING WITH THE SIMULATION #############
####################################################

set.seed(123456)
fine_results_list <- NULL

for (P in Pset) {

  for (ds in 1:length(datasets)) {

    filename = datasets[ds]
    testfilename = testdatasets[ds]
    results = NULL

    countrepetitions = 0
    for (aaa in 1:2000) {

      try({
        dataset = load.dataset(filename, NSET, NTRAIN, testfilename)
        npartition = P * nreplicates

        nclasses = length(unique(dataset$trainclasses))


        partitions = NULL
        if (is_balanced) {
          partitions = create.random.partition(dataset$trainset, dataset$trainclasses, npartition)
        } else {
          partitions = create.perturbated.partition(dataset$trainset, dataset$trainclasses, npartition)
        }


        lapply(partitions$classes, function(l) length(unique(l)))

        w = NULL
        wreverse = NULL
        distances = rep(0, npartition)

        resultsbyclassifier = list(list(NULL))

        for (clasif in 1:length(classifiers)) {
            resultsbyclassifier[[clasif]] = replicate(P,list())
        }

        for (p in 1:length(partitions$partitions)) {
          #direct weights
          w[[p]] = energy.weights.sets(partitions$partitions[[p]], dataset$testset, bound=4)
          #reverse weights
          wreverse[[p]] = energy.weights.sets(dataset$testset, partitions$partitions[[p]], bound=4)
          #energy distance
          distances[p] = energy.stat(partitions$partitions[[p]], dataset$testset)
          #node classification
          for (ci in 1:length(classifiers)) {
            resultsbyclassifier[[ci]][[p]] = classifiers[[ci]](partitions$partitions[[p]], partitions$classes[[p]],
                                                              dataset$testset, dataset$testclasses)$prob

          }
        }

        #save the whole fine grained results for later analysis
        fine_results_list <- append( fine_results_list,
                                     list(probs_by_classifier = resultsbyclassifier,
             distances = distances,
             wreverse = wreverse,
             w = w,
             true_classes = dataset$testclasses,
             P = P,
             ds = ds))

        rules.by.classif = NULL
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

          resline = c(
            mean(sapply(beliefs, sum.rule) == dataset$testclasses),
            mean(sapply(beliefs, product.rule) == dataset$testclasses),
            mean(sapply(beliefs, majority.rule) == dataset$testclasses),
            mean(sapply(beliefs, min.rule) == dataset$testclasses),
            mean(sapply(beliefs, max.rule) == dataset$testclasses),

            mean(sapply(distbeliefs, sum.rule) == dataset$testclasses),
            mean(sapply(distbeliefs, product.rule) == dataset$testclasses),
            mean(sapply(distbeliefs, majority.rule,
                        weights = (1 / distances) /  sum(1 / distances) ) == dataset$testclasses),
            mean(sapply(distbeliefs, min.rule) == dataset$testclasses),
            mean(sapply(distbeliefs, max.rule) == dataset$testclasses),


            mean(sapply(reversebeliefs, sum.rule) == dataset$testclasses),
            mean(sapply(reversebeliefs, product.rule) == dataset$testclasses),
            mean( reverse.majority.rule(beliefs, wreverse) == dataset$testclasses),
            mean(sapply(reversebeliefs, min.rule) == dataset$testclasses),
            mean(sapply(reversebeliefs, max.rule) == dataset$testclasses)
          )


          rules.by.classif = rbind(rules.by.classif, resline)
        }

        results = c(results, list(rules.by.classif))
        print(aaa)

      })
      countrepetitions = countrepetitions + 1
      if (countrepetitions == NREP) break
    }


    resmeans = Reduce("+" , results) / length(results)

   # ressd = sqrt(Reduce( "+", lapply(results, function (x) (x- resmeans)**2)) / length(results))

    resline = cbind(filename, P, namesclassifiers, resmeans)
    colnames(resline) <- c("DATASET", "P", "CLASSIFIER",
                           "BASE.SUM", "BASE.PROD", "BASE.MAJ", "BASE.MIN", "BASE.MAX",
                           "DIST.SUM", "DIST.PROD", "DIST.MAJ", "DIST.MIN", "DIST.MAX",
                           "REV.SUM", "REV.PROD", "REV.MAJ", "REV.MIN", "REV.MAX"
                           )

    totalresults = rbind(totalresults, resline)

    save(totalresults, file = "5classifiers_unbal_decisionrule.RData")
    save(fine_results_list, file= "5class_unbal_fine_grained.RData")

  }


}

########################################################
########## RUN THE GLOBAL RESULTS ######################
########################################################

set.seed(123456)
global_results_list <- NULL
for (ds in 1:length(datasets)) {
  for (nrep in 1:NREP) {
    print(nrep)
  filename = datasets[ds]
  testfilename = testdatasets[ds]
  dataset = load.dataset(filename, NSET, NTRAIN, testfilename)
  resultsbyclassifier <- list()
  for (ci in 1:length(classifiers)) {
    resultsbyclassifier[[ci]] = classifiers[[ci]](dataset$trainset, dataset$trainclasses,
                                                       dataset$testset, dataset$testclasses)$prob
  }
  global_results_list <- append( global_results_list,
                               list(probs_by_classifier = resultsbyclassifier,
                                    filename=filename,
                                    classifiers=namesclassifiers,
                                    test_classes <- dataset$testclasses))
  }

}
save(global_results_list, file="5class_global.RData")

quit()
