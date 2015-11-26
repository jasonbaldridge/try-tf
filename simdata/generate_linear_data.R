### 10/29/2015
### Simulate a two-class linearly separable classification problem.
### Label 0 is the "negative" class.
### Label 1 is the "positive" class.
### Author: Jason Baldridge

# Create a matrix given a label, the class means of some number of
# dimensions, the number of items, and the standard deviation. Values
# are sampled normally according to the mean and stdev for each
# column.
create_matrix = function(label, mu, n, dev=.1) {
  d = length(mu)
  x = t(matrix(rnorm(n*d, mu, dev), ncol=n))
  cbind(rep(label,n),x)
 }

# Num input dimensions (the "features").
numDimensions = 2
  
# Sample the means for the dimensions for a positive class.
#pos = runif(numDimensions,min=0,max=1)
pos = c(.7,.5) # Use a fixed 2-dimensional center.
  
# Sample the means for the dimensions for a negative class.
#neg = runif(numDimensions,min=0,max=1)
neg = c(.3,.1) # Use a fixed 2-dimensional center.

# Create training data.
numTraining = 500
trainDev = .1
training_data = as.matrix(rbind(create_matrix(1,pos,numTraining,trainDev),create_matrix(0,neg,numTraining,trainDev)))
shuffled_training_data = training_data[sample(nrow(training_data)),]
write.table(shuffled_training_data,file="linear_data_train.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

# Create eval data. Possibly make the stdev bigger to make it a bit more interesting.
numEval = 100
evalDev = .1
eval_data = as.matrix(rbind(create_matrix(1,pos,numEval,evalDev),create_matrix(0,neg,numEval,evalDev)))
shuffled_eval_data = eval_data[sample(nrow(eval_data)),]
write.table(shuffled_eval_data,file="linear_data_eval.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

#Plot the training items, if desired.
#colnames(training_data) = c("label","x","y")
#plot(training_data[,c("x","y")],pch=21,bg=c("orange","blue")[training_data[,"label"]+1])

