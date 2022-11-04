gene1 <- c(10,11,8,3,2,1)
gene2 <- c(6,4,5,3,2.8,1)
k <- c('mouse1','mouse2','mouse3', 'mouse4','mouse5','mouse6')
data <- data.frame(gene1,gene2,row.names = k )
data
plot(data)

summary(data)

t <- t(data)
gene.pc<- prcomp(data, center = TRUE, scale. = TRUE)

# library(ggbiplot)

ggbiplot(gene.pc)

ggbiplot(gene.pc, labels=rownames(data))


plot(gene.pc$x[,1], gene.pc$x[,2])

###########################

library(MASS)
mat <- matrix(c(3,-1,1,3,1,1), nrow = 2, ncol = 3)
mat

svd(mat)
