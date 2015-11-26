### 10/29/2015
### Simulate a two-class "Saturn" classification problem
### Label 0 is the planet
### Label 1 is the ring
### Author: James Scott
  
# @n: number of points
# @frac: fraction of points to simulate from class 1
# @d: Euclidean dimension
# @radius: a 2-vector of radiuses for class0 and class1
sim_saturn_data = function(n, d, radius, sigma, frac = 0.5) {
	
	# Argument checking
	stopifnot(d >= 2, length(radius) == 2)
	
	# We work in radial coordinates.
	# Uniformly sample d-1 angular coordinates for each point
	phi = matrix(runif(n*(d-1), 0, 2*pi), nrow=n, ncol=d-1)
	
	# Sample a class indicator for each simulated data point
	gamma = rbinom(n, 1, frac)
	n1 = sum(gamma)
	
	# Simulate a radial distance for each point
	r = rep(0, n)
	r[gamma==0] = runif(n-n1, 0, radius[1])
	r[gamma==1] = rnorm(n1, radius[2], sigma)

	# convert to Euclidean coordinates
	x = matrix(0, nrow=n, ncol=d)
	x[,1] = r*cos(phi[,1])
	x[,d] = r*apply(sin(phi), 1, prod)
	if(d >= 3) {
		for(j in 2:(d-1)) {
			prod_of_sines = apply(matrix(sin(phi[,1:(j-1)]), nrow=n), 1, prod)
			x[,j] = r*prod_of_sines*cos(phi[,j])
		}
	}
	
	list(labels = gamma, features = x)
}

### Testing: simulate some data and plot it.
mycols = c('blue','orange')

# 2d example
#out = sim_saturn_data(1000, 2, c(3, 10), sigma = 1)
#plot(out$features,pch=21,bg=mycols[out$labels+1],xlab="x",ylab="y")
  
# 3d example (need rgl installed for the visualization)
#out = sim_saturn_data(1000, 3, c(3, 10), sigma = 1.0)  
#rgl::plot3d(out$features, col=mycols[out$labels+1],xlab="x",ylab="y",zlab="z")

### Actually create simulated data.
numDimensions = 2
    
# Create training data.
numTraining = 500
training_out = sim_saturn_data(numTraining, numDimensions, c(5, 10), sigma = 1.0)
training_data = cbind(training_out$labels,training_out$features)
write.table(training_data,file="saturn_data_train.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

# Create eval data. Perhas make sigma bigger to make it a bit more interesting.
numEval = 100
eval_out = sim_saturn_data(numEval, numDimensions, c(5, 10), sigma = 1.0)
eval_data = cbind(eval_out$labels,eval_out$features)
write.table(eval_data,file="saturn_data_eval.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

