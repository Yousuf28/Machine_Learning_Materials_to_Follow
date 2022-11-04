# https://stats.idre.ucla.edu/r/codefragments/svd_demos/


library(foreign)
auto <- read.dta("http://statistics.ats.ucla.edu/stat/data/auto.dta")

pca.m1 <- prcomp(~trunk + weight + length + headroom, data = auto,
                 scale = TRUE)

screeplot(pca.m1)
summary(pca.m1)

# library(ggbiplot)
ggbiplot(pca.m1)

ggbiplot(pca.m1, labels=rownames(auto))
head(auto)
################

xvars <- with(auto, cbind(trunk, weight, length, headroom))
corr <- cor(xvars)
a <- eigen(corr)
(std <- sqrt(a$values))
## [1] 1.738 0.807 0.526 0.225
(rotation <- a$vectors)
##        [,1]   [,2]   [,3]    [,4]
## [1,] -0.507 -0.233  0.825  0.0921
## [2,] -0.522  0.454 -0.268  0.6708
## [3,] -0.536  0.390 -0.137 -0.7358
## [4,] -0.428 -0.767 -0.479 -0.0057
# svd approach
df <- nrow(xvars) - 1
zvars <- scale(xvars)
z.svd <- svd(zvars)
z.svd$d/sqrt(df)
## [1] 1.738 0.807 0.526 0.225
z.svd$v
##       [,1]   [,2]   [,3]    [,4]
## [1,] 0.507 -0.233  0.825 -0.0921
## [2,] 0.522  0.454 -0.268 -0.6708
## [3,] 0.536  0.390 -0.137  0.7358
## [4,] 0.428 -0.767 -0.479  0.0057