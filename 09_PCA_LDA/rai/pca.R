# Data
data("iris")
str(iris)
summary(iris)

# Partition Data
set.seed(111)
ind <- sample(2, nrow(iris),
              replace = TRUE,
              prob = c(0.8, 0.2))
training <- iris[ind==1,]
testing <- iris[ind==2,]

# Scatter Plot & Correlations
library(psych)
pairs.panels(training[,-5],
             gap = 0,
             bg = c("red", "yellow", "blue")[training$Species],
             pch=21)

# Principal Component Analysis
pc <- prcomp(training[,-5],
             center = TRUE,
             scale. = TRUE)
           
attributes(pc)
pc$center
pc$scale
print(pc)
summary(pc)

# Orthogonality of PCs
pairs.panels(pc$x,
             gap=0,
             bg = c("red", "yellow", "blue")[training$Species],
             pch=21)

# Bi-Plot
library(devtools)
# install_github("ggbiplot", "vqv")
library(ggbiplot)
g <- ggbiplot(pc,
              obs.scale = 1,
              var.scale = 1,
              groups = training$Species,
              ellipse = TRUE,
              circle = TRUE,
              ellipse.prob = 0.68)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal',
               legend.position = 'top')
print(g)

# Prediction with Principal Components
trg <- predict(pc, training)
trg <- data.frame(trg, training[5])
tst <- predict(pc, testing)
tst <- data.frame(tst, testing[5])

# Multinomial Logistic regression with First Two PCs
library(nnet)
trg$Species <- relevel(trg$Species, ref = "setosa")
mymodel <- multinom(Species~PC1+PC2, data = trg)
summary(mymodel)

# Confusion Matrix & Misclassification Error - training
p <- predict(mymodel, trg)
tab <- table(p, trg$Species)
tab
1 - sum(diag(tab))/sum(tab)

# Confusion Matrix & Misclassification Error - testing
p1 <- predict(mymodel, tst)
tab1 <- table(p1, tst$Species)
tab1
1 - sum(diag(tab1))/sum(tab1)
