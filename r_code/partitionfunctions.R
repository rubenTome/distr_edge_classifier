####################################################################
#### distances between distributions ###############################
####################################################################

#X e Y son las matrices con los datos, cada fila un dato
energy.stat = function (X , Y ) {
  # X = as.matrix(X) #transformar a matriz por si es un vector unidimensional
  # Y = as.matrix(Y)
  # d = dist(rbind(X,Y)) #distancias euclideas para todos los pares posibles de osbervaciones
  # N = nrow(X)
  # M = nrow(Y)
  # d = as.matrix(d) #transforma el objeto tipo dist a una matriz cuadrada simetrica que contiene las distancias
  # #division en within sample y between sample groups y calculo del estadistico
  # stat = 2*mean(d[1:N, (1:M) + N]) - mean( d[1:N, 1:N]) - mean( d[(1:M) + N, (1:M) + N])
  # stat * (N*M) / (N + M)
  X = as.matrix(X)
  Y = as.matrix(Y)
  energy::eqdist.e(rbind(X,Y), c(nrow(X), nrow(Y))) / var(as.vector(rbind(X,Y)))
  #energy::eqdist.etest(rbind(X,Y), c(nrow(X), nrow(Y)), R=10000)
}


MMD.stat = function (X, Y) {
  X = as.matrix(X)
  Y = as.matrix(Y)
  d = dist(rbind(X,Y))
  
  N = nrow(X)
  M = nrow(Y)
  d = as.matrix(d)
  
  gamm = median(d) #calculamos la ventana para el kernel gaussiano con la heuristica de la mediana
  
  kernel = exp((-1/gamm**2)*(d^2)) #aplicar kernel gaussiano
  
  mean(kernel[1:N, 1:N]) + mean(kernel[(1:M) + N, (1:M) + N]) - 2*mean(kernel[1:N, (1:M) + N])
}


DD.stat = function(X, Y) {
  X = as.matrix(X)
  Y = as.matrix(Y)
  d = dist(rbind(X,Y))
  
  N = nrow(X)
  M = nrow(Y)
  d = as.matrix(d)
  
  dW = c(d[1:N, 1:N], d[(1:M) + N, (1:M) + N]) #distancias within sample por un lado
  dB = c(d[1:N, (1:M) + N]) #distancias between sample por otro lado
  
  # a partir de aqui se calcula cramervon mises muestral mediante la formula cerrada de anderson
  # usando las distancias within y between como muestras a comparar (la formula esta en la wikipedia: Cramer vonMises criterion (twosample))
  N = length(dW)
  M = length(dB)
  comb = c(dW,dB) #se vuelve a unir
  
  ord = rank(comb); #se obienen los rangos
  
  rankX = sort(ord[1:N]); #se ordenan los rangos dentro de cada grupo
  rankY = sort(ord[(N+1):(N+M)])
  
  
  sr = N*sum((rankX - 1:N)**2)
  ss = M*sum((rankY - 1:M)**2)
  
  
  U = ss + sr
  N = as.double(N)
  M = as.double(M)
  stat = U/(N*M*(N+M)) - (4*M*N-1)/(6*(M+N))
  stat
  
}



hellinger.dist = function( X, Y) {
  X = as.vector(X)
  Y = as.vector(Y)
  uniq = unique(c(X,Y))
  countX = sapply(uniq, function(x) {
    sum(X == x)
  })
  countX = countX / sum(countX)
  
  countY = sapply(uniq, function(x) {
    sum(Y == x)
  })
  countY = countY / sum(countY)
  
  sum((sqrt(countX) - sqrt(countY))**2)/ 2.0
  
  # PER COLUMN DIST
  # freqtabl = function(dataset) {
  #   apply(dataset, 2, function (colu) {
  #     counts = sapply(uniq, function(x) {
  #       sum(colu == x)
  #     })
  #     counts / sum(counts)
  #   })
  # }
  
}

hamming.dist = function( X, Y) {
  acum = 0
  for (i in 1:nrow(X)) {
    for (j in 1:nrow(Y)) {
      acum = acum +mean(X[i,] - Y[j,] != 0)
    }
  }
  acum
}

library(flexclust)
euclidiff.dist = function(X,Y) {
  sum(dist2(X,Y))
}

nn.dist = function(X,Y) {
  sum( apply(dist2(X,Y) , 1, min) )
}

oracle.dist = function (X, Y, labelX, labelY) {
  oracleknn = class::knn(X, Y, labelX)
  1 - mean(oracleknn == labelY)
}


##########################################################################


load.dataset = function(filename, maxsize, trainsize, testfilename = "") {
  dataset = read.csv(filename, sep=",")
  
  samp = sample(nrow(dataset))[1:maxsize] #shuffle the data
  #divide into train and test sets
  trainset = dataset[ samp[1:trainsize], ]
  trainclasses = trainset[,ncol(trainset)]
  trainset = as.matrix(trainset[, -ncol(trainset)])
  testset = dataset[ samp[-(1:trainsize)], ]
  testclasses = testset[,ncol(testset)]
  testset = as.matrix(testset[, -ncol(testset)])
  
  if (testfilename != "") { #different test file
    testdataset = read.csv(testfilename, sep=",")
    testset = testdataset[ sample(nrow(testdataset), maxsize - trainsize) ,] #get a subset of the testset
    testclasses = testset[,ncol(testset)]
    testset = as.matrix(testset[, -ncol(testset)])
  }
  
  
  #normalize classes to integers starting in 1
  CL = as.numeric(as.factor(c(trainclasses, testclasses)))
  trainclasses = CL[1:length(trainclasses)]
  testclasses = CL[ -(1:length(trainclasses))]
  
  colnames(trainset) <- NULL
  colnames(testset) <- NULL
  
  
  list(trainset=trainset, trainclasses = trainclasses,
       testset=testset, testclasses = testclasses)
}

create.random.partition = function(trainset, trainclasses, npartitions) {
  
  C = unique(trainclasses)
  
  #divide the trainset by class
  CLASSset = sapply( C , function (x) {
    whichmyclass = which(trainclasses == x)
    selected = sample(whichmyclass)
    if (length(whichmyclass) == 1) {
      selected = whichmyclass
      list( t(as.matrix(trainset[ selected , ])))
    } else {
    list( as.matrix(trainset[ selected , ]))
    }
  })
  
  CLASSrows = lapply(CLASSset, function (x) max( floor(nrow(x)/npartitions), 1) )
  
  CLASSpart = lapply( CLASSset, function(x) {
    myrows = floor(nrow(x)/npartitions)  #calculate the number of elements per class
    droprep = FALSE
    if (myrows == 0) { #strange case
      myrows = 1
      x = x[ sample(nrow(x), myrows*npartitions, rep=TRUE),]
    }
    #divide the elements of each class into n partitions
    split(data.frame(as.matrix(x[1:(myrows*npartitions), ])), rep(1:npartitions, myrows))
  })
  
  #merge all perclass partitions into the final partitions containing elements from each class
  partitions = lapply( 1:npartitions, function(i) {
    parti = NULL
    for (j in 1:length(C)) {
      parti = rbind(parti, as.matrix(CLASSpart[[j]][[i]]))
    }
    colnames(parti) <- NULL
    parti
  })
  partitionclasses = rep(C, unlist(CLASSrows))
  partitionclasses = sapply(1:npartitions, function(i) list(partitionclasses))
  
  list(partitions = partitions, classes = partitionclasses) 
}

boosting.partition = function(trainset, trainclasses, npartitions, replicates) {
  partitions = NULL
  classes = NULL
  for (i in 1:replicates) {
    p = create.random.partition(trainset, trainclasses, npartitions)
    classes = p$classes
    partitions = c(partitions, p$partitions)
  }
  
  list(partitions = partitions, classes = classes) 
}


#C number of classes
#P number of partitions
#N number of observations in the set
#S strength of the perturbation
perturbation.whole = function(S, N, C, P) {
  #perturbation, 10% of the observations are changed
  dev  = rep(0,C)
  D = ceiling((N*S)/(P))
  
  R = D
  
  for (i in 1:(C-1)) {
    sampled.obs = sample(R,1)
    dev[i] = sampled.obs
    R = R - sampled.obs
    if (R == 0) break
  }
  dev[C] = dev[C] + R
  sample(dev)
}



create.perturbated.partition = function(trainset, trainclasses, npartitions) {
  
  #1) calculate proportions per class
  #2) slightly perturbate these proportions
  #3) create a partition with these proportions, if possible
  #4) go to to step 1
  
  #at least one observation per class
  
  remainingset = trainset
  remainingclasses  = trainclasses
  C = length(unique(remainingclasses))
  #cap to one at minimum
  
  table(remainingclasses)
  
  partitions = list()
  partitionclasses = list()
  
  for (i in 1:(npartitions-1)) {
    
    N = length(remainingclasses)
    
    P = npartitions - i + 1
    
    prop = table(remainingclasses) / N
    
    #perturbation, change each proportion by up to 70%
    dev = prop * runif(C, 0.1, 1.9)
    dev = dev / sum(dev)  #normalize the proportions
    
    if ((i == 1) ) { #some partitions have the same proportion as the original
      dev = prop
    }

    observations = floor(dev * (N / P)) #calculate how many observations in each 

    partitions[[i]] = numeric(0)
    partitionclasses[[i]] = numeric(0)
    for (j in 1:C) {
      
      rem = which(remainingclasses == j)

      if (length(rem) == 0) {
        print(paste("ERROR NO ELEMENTS  OF CLASS", j))
        stop()
      }

      nobs = observations[j]
      
      if (nobs == 0) { #at least one observation per class
        nobs=1
      }
 
      nremclass = length(rem)

      nobs = min(nobs, nremclass)

      selectedobs = sample(rem, nobs)
      if (length(rem) == 1) { #this trick because sample works different when only one number is given as first argument
        selectedobs = rem
      }
      partitions[[i]] = rbind( partitions[[i]], 
                                 remainingset[ selectedobs, ])
      partitionclasses[[i]] = c(partitionclasses[[i]], remainingclasses[selectedobs])

      #do not remove all remaining by adding one more of what we are going to take

      if ((table(remainingclasses)[j] - nobs) < 1) {

        toadd = nobs
        remainingset = rbind(remainingset, remainingset[rem[1:toadd],])
        remainingclasses = c(remainingclasses, remainingclasses[ rem[1:toadd] ])
      }
      
      remainingset = remainingset[ -(selectedobs), ]
      remainingclasses = remainingclasses[ -(selectedobs) ]

    }

  }

  partitions[[npartitions]] = remainingset
  partitionclasses[[npartitions]] = remainingclasses
  
  list(partitions=partitions, classes = partitionclasses)
}


library(kernlab)

energy.weights.sets = function(trainset, testset, bound=4) {
  
  n = nrow(trainset) # the number of data points
  
  d = ncol(trainset)
  
  distances = as.matrix( dist(rbind(trainset, testset)) )

  #in energy minimize: 2* mean(BETWEEN) - mean(WITHIN.A) - mean(WITHIN.B)
  distances = exp(-distances)

  K = distances[1:nrow(trainset), 1:nrow(trainset)]
  k = distances[1:nrow(trainset), 1:nrow(testset) + nrow(trainset)]
  WB = distances[1:nrow(testset) + nrow(trainset), 1:nrow(testset) + nrow(trainset)]
  k = rowMeans(k)
  
  #the constanst for the constatint, b0 in the funcion
  
  B = 1
  m = n
  

  c = -k  #we do multiply by 2 because the equation in kernlab::ipop is already halved
  H = K
  
  A = matrix(0, n, n)
  A[1,] = 1
  
  b = rep(0, n)
  r = rep(1, n)
  
  l = rep(0, n)
  u = rep(1, n)

  sol = ipop(c, H, A, b, l, u, r, sigf=4, maxiter = 45, bound=bound, margin = 0.01, verb=FALSE)
  
  x = primal(sol)

  
  #x = rep(1/nrow(trainset), nrow(trainset))
  #x %*% -K %*% x + k %*% x
  
  list(weights=x, val=-2*k %*% x + x %*% K %*% x + mean(WB))
}


kfun = function(x,y) {
  
  -sum((x-y)**2) + sum(x**2) + sum(y**2)
}

# KKK = proxy::dist(rbind(trainset, testset),method= kfun)
# KKK = as.matrix(KKK)

#just in case the other approach is not valid
lbfgs.energyweight =  function(trainset, testset, bound=4) {
  
  n = nrow(trainset) # the number of data points
  
  d = ncol(trainset)
  
  distances = as.matrix( dist(rbind(trainset, testset)) )
  
  #in energy minimize: 2* mean(BETWEEN) - mean(WITHIN.A) - mean(WITHIN.B)
  
  WA = distances[1:nrow(trainset), 1:nrow(trainset)]
  WB = distances[1:nrow(testset) + nrow(trainset), 1:nrow(testset) + nrow(trainset)]
  BETWEEN= distances[1:nrow(trainset), 1:nrow(testset) + nrow(trainset)]
  BETWEEN = rowMeans(BETWEEN)
  
  #lagrangian approach
  
  mylagrange = function(lambda) {
  
  myopt = function(w, lambda) {
    f = (2 * BETWEEN %*% w - w %*% WA %*% w - mean(WB))
    g = sum(w)  -  1
    f + lambda*g
  }
  
  w = runif(nrow(trainset), 0, 1) #rep(1, nrow(trainset), nrow(trainset)) #
  w = w / sum(w)

  opt = optim(w, myopt, lambda=lambda, method="L-BFGS-B", lower=rep(0, nrow(trainset) ),
        upper = rep(1, nrow(testset)),control =list(trace=1, factr=1e10))
  -opt$val
  }
  lam = runif(1)
  opt = optim(lam, mylagrange, method="L-BFGS-B", lower=0 ,
              upper = 1000,control =list(trace=1, factr=1e10))
  
  
  lam = seq(0.07163631, 0.09181818, length.out=100)
  vals = lam
  ind = 1
  len = 100
  mini = lam[1]
  maxi = lam[100]
  while (TRUE) {
    midi = (mini + maxi) / 2.0
    vmid = mylagrange(midi)
    print(paste("GOMID ", midi, " " , vmid))
    if (vmid < vmax) {
      maxi = midi
      vmax = mylagrange(maxi)
    } else {
      mini = midi
      vmin = mylagrange(mini)
    }
  }
  
  
  w  = opt$par
  list(weights=opt$par, val= 2 * BETWEEN %*% w - w %*% WA %*% w - mean(WB) )
}

# dataset = load.dataset("scenariosimul/scenariosimulC2D3G3STDEV0.05.csv", 1000, 500)
# energy::eqdist.e( rbind(dataset$trainset, dataset$testset),
#                   c(nrow(dataset$trainset),nrow(dataset$trainset)) )
# 
# gradweig = lbfgs.energyweight( dataset$trainset, dataset$testset)
# sum(abs(w - gradweig$weights))
# w = gradweig$weights
# 
# round(gradweig$weights, 4)
# sum(gradweig$weights)
# 
# quadweig = energy.weights.sets(dataset$trainset, dataset$testset)
# 
# sum(abs(gradweig$weights - quadweig$weights))
# 
# 
# trainset = dataset$trainset
# testset = dataset$testset
# distances = as.matrix( dist(rbind(trainset, testset)) )
# #in energy minimize: 2* mean(BETWEEN) - mean(WITHIN.A) - mean(WITHIN.B)
# #distances = exp(-distances)
# 
# K = distances[1:nrow(trainset), 1:nrow(trainset)]
# k = distances[1:nrow(trainset), 1:nrow(testset) + nrow(trainset)]
# WB = distances[1:nrow(testset) + nrow(trainset), 1:nrow(testset) + nrow(trainset)]
# k = rowMeans(k)
# 
# w = quadweig$weights
# 2*k %*% w - w %*% K %*% w - mean(WB)
# w = gradweig$weights
# 2*k %*% w - w %*% K %*% w - mean(WB)
# w = rep(1/length(w), length(w))
# 2*k %*% w - w %*% K %*% w - mean(WB)


# knn.classifier.results = function(trainset, trainclasses, testset, testclasses) {
#   predclasses = class::knn(trainset, testset, trainclasses)
#   accu = mean(predclasses == testclasses)
#   list(classes = predclasses, accu = accu)
# }
# 
# svm.classifier.results = function(trainset, trainclasses, testset, testclasses) {
#   SVM = e1071::svm(x=trainset, y=as.factor(trainclasses), scale=FALSE)
#   predclasses = ( predict(SVM, testset) )
#   accu=mean(predclasses == as.factor(testclasses))
#   list(classes = predclasses, accu = accu)
# }
# 
# forest.classifier.results = function(trainset, trainclasses, testset, testclasses) {
#   forestmodel = randomForest::randomForest(x=trainset, y=as.factor(trainclasses))
#   predclasses = as.numeric(predict(forestmodel, testset))
#   accu = mean(predclasses == as.factor(testclasses))
#   list(classes = predclasses, accu = accu)
# }
# 
# naive.classifier.results = function(trainset, trainclasses, testset, testclasses) {
#   model <- e1071::naiveBayes(x=trainset, y=as.factor(trainclasses))
#   predclasses = predict(model, testset)
#   accu=mean(predclasses == testclasses)
#   list(classes = predclasses, accu = accu)
# }
#   