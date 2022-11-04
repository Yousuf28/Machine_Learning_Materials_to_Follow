
####
##  what PCA does?
## it show correlatin of variables (columns)
## cluster the observation (rows)



head(mtcars)
str(mtcars)
View(mtcars)

mtcars.pca <- prcomp(mtcars[, c(1:7, 10,11) ], center = TRUE, scale. = TRUE)

summary(mtcars.pca)
library(ggbiplot)

ggbiplot(mtcars.pca)


ggbiplot(mtcars.pca, labels=rownames(mtcars))


mtcars.country <- c(rep("Japan", 3), rep("US",4), rep("Europe", 7),rep("US",3), "Europe", rep("Japan", 3), rep("US",4), rep("Europe", 3), "US", rep("Europe", 3))


ggbiplot(mtcars.pca,ellipse=TRUE, 

         labels=rownames(mtcars), groups=mtcars.country)
