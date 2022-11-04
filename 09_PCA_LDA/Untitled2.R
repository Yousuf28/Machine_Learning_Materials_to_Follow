library(ggplot2)

cars <- mtcars
cars$cyl <- factor(cars$cyl, labels = 
                       c('Four cylinder', 'Six cylinder', 'Eight cylinder'))

features <- c('wt', 'qsec')
n_clusters <- 3
car_clusters <- kmeans(cars[, features], n_clusters, nstart = 30)

cars$cluster <- factor(car_clusters$cluster)

centroids <- data.frame(cluster = factor(seq(1:n_clusters)),
                        wt = car_clusters$centers[,'wt'],
                        qsec = car_clusters$centers[,'qsec'])

# cross tab of cylinder by cluster
print(table(cars$cluster, cars$cyl))

g <- ggplot() + 
    geom_point(data = cars, 
               aes(x = wt, 
                   y = qsec,
                   color = cluster),
               size = 3) +
    geom_text(data = cars,
              aes(x = wt,
                  y = qsec,
                  label = row.names(cars),
                  color = cluster),
              nudge_y = .2,
              check_overlap = TRUE) +
    geom_point(data = centroids,
               mapping = aes(x = wt,
                             y = qsec,
                             color = cluster),
               size = 20,
               pch = 13) 

print(g)