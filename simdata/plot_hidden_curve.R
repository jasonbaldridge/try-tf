d = read.table("output_curve_hidden_nodes.txt",sep=",")
d$V2 = d$V2*100
means = aggregate(V2 ~ V1, d, mean)

#jpeg("hidden_node_curve.jpg")  
plot(means$V1,means$V2,xlab="Number of hidden nodes.",ylab="Accuracy",ylim=c(40,100),pch=21,bg="blue")
lines(means$V1,means$V2)
points(d$V1,jitter(d$V2,1),pch=1,cex=.5)
#dev.off()