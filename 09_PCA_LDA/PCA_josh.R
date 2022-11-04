
# https://statquest.org/2017/11/27/statquest-pca-in-r-clearly-explained/#code

data.matrix <- matrix(nrow=100, ncol=10)
colnames(data.matrix) <- c(
    paste("wt", 1:5, sep=""),
    paste("ko", 1:5, sep=""))
rownames(data.matrix) <- paste("gene", 1:100, sep="")
for (i in 1:100) {
    wt.values <- rpois(5, lambda=sample(x=10:1000, size=1))
    ko.values <- rpois(5, lambda=sample(x=10:1000, size=1))
    
    data.matrix[i,] <- c(wt.values, ko.values)
}

tdata <- t(data.matrix)

head(tdata)

head(data.matrix)

dim(data.matrix)
pca <- prcomp(t(data.matrix), scale=TRUE) 

summary(pca)








library(ggbiplot)
ggbiplot(pca)

ggbiplot(pca, labels=rownames(tdata))

## plot pc1 and pc2
plot(pca$x[,1], pca$x[,2])
## make a scree plot
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per, main="Scree Plot", xlab="Principal Component", ylab="Percent Variation")

library(ggplot2)

pca.data <- data.frame(Sample=rownames(pca$x),
                       X=pca$x[,1],
                       Y=pca$x[,2])
pca.data

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
    geom_text() +
    xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
    ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) +
    theme_bw() +
    ggtitle("My PCA Graph")


loading_scores <- pca$rotation[,1]

gene_scores <- abs(loading_scores) ## get the magnitudes
gene_score_ranked <- sort(gene_scores, decreasing=TRUE)
top_10_genes <- names(gene_score_ranked[1:10])

top_10_genes ## show the names of the top 10 genes

pca$rotation[top_10_genes,1] ## show the scores (and +/- sign)


tdata <- t(data.matrix)
heatmap(tdata)
