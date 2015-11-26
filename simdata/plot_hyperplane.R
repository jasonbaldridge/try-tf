filename = "linear_data_train"

data = read.table(paste(filename,".csv",sep=''),header=F,sep=',')
colnames(data) = c("label","x","y")

#jpeg("linear_data_hyperplane.jpg")
plot(data[,c("x","y")],pch=21,bg=c("orange","blue")[data[,"label"]+1],xlim=c(-.25,1),ylim=c(-.25,1))
abline(h=0,lty=2)
abline(v=0,lty=2)

w0 = -1.87038445
w1 = -2.23716712
b = 1.57296884
slope = -1*(w0/w1)
intercept = -1*(b/w1)
abline(coef=c(intercept,slope),lwd=3)
#dev.off()