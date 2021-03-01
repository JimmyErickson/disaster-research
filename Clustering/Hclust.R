#prep data
library(dendextend)
disasters <- read.csv("z:/Desktop/Research Data India/Spring-2021/Datasets/Timeline_Data_Output_Trimmed.csv",TRUE,",")

str(disasters)
summary(disasters)
any(is.na(disasters))

totalEvents <- disasters$events
sigEvents <- disasters$significantEvents

tEvents <- data.frame(disasters)
sEvents <- data.frame(disasters)

tEvents$events <- NULL
sEvents$significantEvents <- NULL

tEvents_sc <- as.data.frame(scale(tEvents))
sEvents_sc <- as.data.frame(scale(sEvents))

t_dist <- dist(tEvents_sc, method = 'euclidean')
s_dist <- dist(sEvents_sc, method = 'euclidean')

t_hclust_avg <- hclust(t_dist, method = 'average')
s_hclust_avg <- hclust(s_dist, method = 'average')

plot(t_hclust_avg)
rect.hclust(t_hclust_avg , k = 4, border = 2:6)
abline(h = 4, col = 'red')
plot(s_hclust_avg)

t_cut_avg <- cutree(t_hclust_avg, k = 4)
s_cut_avg <- cutree(s_hclust_avg, k = 4)


avg_dend_obj <- as.dendrogram(t_hclust_avg)
avg_col_dend <- color_branches(avg_dend_obj, h = 4)
plot(avg_col_dend)