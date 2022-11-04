

data.matrix <- matrix(nrow=30, ncol=10)
colnames(data.matrix) <- c(
    paste("wt", 1:5, sep=""),
    paste("ko", 1:5, sep=""))
rownames(data.matrix) <- paste("gene", 1:30, sep="")
for (i in 1:30) {
    wt.values <- rpois(5, lambda=sample(x=10:1000, size=1))
    ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))
    
    data.matrix[i,] <- c(wt.values, ko.values)
}

tdata <- t(data.matrix)
View(tdata)

gene.pca<- prcomp(tdata, center = TRUE, scale. = TRUE)

library(ggbiplot)

ggbiplot(gene.pca)

ggbiplot(gene.pca, labels=rownames(tdata))


gene.type <- c(rep("wt", 5), rep("ko",5))

ggbiplot(gene.pca,ellipse=TRUE, 
         
         labels=rownames(tdata), groups=gene.type)


