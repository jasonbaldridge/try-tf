filename = "linear_data_train"
#filename = "moon_data_train"
#filename = "saturn_data_train"

data = read.table(paste(filename,".csv",sep=''),header=F,sep=',')
colnames(data) = c("label","x","y")

jpeg(paste(filename,".jpg",sep=''))
plot(data[,c("x","y")],pch=21,bg=c("orange","blue")[data[,"label"]+1])
dev.off
