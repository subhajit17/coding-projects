#CODING PAGE

#import liberies 
library(party)
library(psych)
library(dplyr)
library(data.table)
library(ggplot2)
library(plotly)
library(expss)
library(pander)
library(forcats)
library(stringr)
library(caTools)
library(VIM)
library(caret)
require(reshape2)
library(GGally)
library(corrplot)
library(factoextra)
library(gridExtra)
library(C50)
library(highcharter)
library(rpart)
library(e1071)
library(ranger)
library(epiR)
library(randomForest)
library(party)
library(class)
library(kknn) 
library(gbm)
library(ada)
library(c3)

data= read.csv(choose.files())

#The dataset has 33 columns, but one is completely empty, so I removed it...
head(data)
str(data)
data=data[,-33]
str(data)
head(data)

#To design a machine learning algorithm that is able to correctly classify whether the tumor is benign or malignant.

#missing data
missing_values = data %>% summarize_all(funs(sum(is.na(.))/n()))
missing_values 
aggr(data, prop = FALSE, combined = TRUE, numbers = TRUE, sortVars = TRUE, sortCombs = TRUE)

#Target variable
table(data$diagnosis)
prop.table(table(data$diagnosis))*100

#Target variable is a character-type variable, so I convert it into a factor.
data$diagnosis=factor(data$diagnosis, labels=c('B','M'))
prop.table(table(data$diagnosis))*100

#Descriptive analysis
psych::describeBy(data[3:32], group=data$diagnosis)
'in general, malignant diagnoses have higher scores in all variables.'

#Mean
df.m <- melt(data[,-c(1,13:32)], id.var = "diagnosis")
p <- ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Variables") + ylab("")+ guides(fill=guide_legend(title="Group"))
p

#Se
df.m <- melt(data[,-c(1,3:12,23:32)], id.var = "diagnosis")
p <- ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Variables") + ylab("")+ guides(fill=guide_legend(title="Group"))
p

#Worst
df.m <- melt(data[,c(2,23:32)], id.var = "diagnosis")
p <- ggplot(data = df.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis)) + facet_wrap( ~ variable, scales="free")+ xlab("Variables") + ylab("")+ guides(fill=guide_legend(title="Group"))
p


#Correlations
pairs.panels(data[,c(3:12)], method="pearson",
             hist.col = "#1fbbfa", density=TRUE, ellipses=TRUE, show.points = TRUE,
             pch=1, lm=TRUE, cex.cor=1, smoother=F, stars = T, main="Cancer Mean")

pairs.panels(data[,c(13:22)], method="pearson",
             hist.col = "#1fbbfa", density=TRUE, ellipses=TRUE, show.points = TRUE,
             pch=1, lm=TRUE, cex.cor=1, smoother=F, stars = T, main="Cancer SE")

pairs.panels(data[,c(23:32)], method="pearson",
             hist.col = "#1fbbfa", density=TRUE, ellipses=TRUE, show.points = TRUE,
             pch=1, lm=TRUE, cex.cor=1, smoother=F, stars = T, main="Cancer Worst")


w.corr<-cor(data[,c(3:12)],method="pearson")
corrplot(w.corr, order='hclust', method='ellipse',addCoef.col = 'black',type='lower', number.cex = 1,tl.cex = 1, diag=F,tl.col = 'black',tl.srt=15)

w.corr<-cor(data[,c(13:22)],method="pearson")
corrplot(w.corr, order='hclust', method='ellipse',addCoef.col = 'black',type='lower', number.cex = 1,tl.cex = 1, diag=F,tl.col = 'black',tl.srt=15)

w.corr<-cor(data[,c(23:32)],method="pearson")
corrplot(w.corr, order='hclust', method='ellipse',addCoef.col = 'black',type='lower', number.cex = 1,tl.cex = 1, diag=F,tl.col = 'black',tl.srt=15)

#We see that there are extremely high correlations between some of the variables

w.corr<-cor(data[,c(3:32)],method="pearson")
corrplot(w.corr, order='hclust', method='ellipse',addCoef.col = 'black',type='lower', number.cex = 0.25,tl.cex = 0.25, diag=F,tl.col = 'black',tl.srt=15)

col<-colorRampPalette(c('blue','white','red'))(20)
heatmap(x=w.corr, col=col,symm=T)

#Training and testing datasets
dataset=data
head(dataset)

#Dataset is divided into two datasets: training (70%) and testing (30%)
set.seed(123)
smp_size <- floor(0.70 * nrow(dataset))
train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)
train <- dataset[train_ind, ]
test <- dataset[-train_ind, ]

#Let's check the target variable.
prop.table(table(train$diagnosis))*100
prop.table(table(test$diagnosis))*100

#K-nn
control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3)
knnFit <- train(diagnosis ~ ., data = train[,-1], method = "knn", trControl = control,tuneLength = 20)
plot(knnFit)

#Testing the model
knnPredict <- predict(knnFit,newdata = test )
cm_knn<-confusionMatrix(knnPredict, test$diagnosis )
cm_knn

#Naive Bayes
#Training the model
learn_nb <- naiveBayes(train[,-c(1,2)], train$diagnosis)
#Testing the model
pre_nb <- predict(learn_nb, test[,-c(1,2)])
cm_nb <- confusionMatrix(pre_nb, test$diagnosis)
cm_nb

#Classification tree
#Training the model
learn_ct <- ctree(diagnosis~., data=train[,-1], controls=ctree_control(maxdepth=2))
#Testing the model
pre_ct   <- predict(learn_ct, test[,-c(1,2)])
cm_ct    <- confusionMatrix(pre_ct, test$diagnosis)
cm_ct

# K-means
#Training the model.

predict.kmeans <- function(newdata, object){
  centers <- object$centers
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}
learn_kmeans <- kmeans(train[,-c(1,2)], centers=2)
#Testing the model.

pre_kmeans <- predict.kmeans(test[,-c(1,2)],learn_kmeans)
pre_kmeans <- factor(ifelse(pre_kmeans == 1,"B","M"))
cm_kmeans <- confusionMatrix(pre_kmeans, test$diagnosis)
cm_kmeans

#Hierarchical
# Dissimilarity matrix
d <- dist(df, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

# Plot the obtained dendrogram
plot(hc1, cex = 0.6, hang = -1)

# Cut tree into 3 groups
sub_grps <- cutree(hc1, k = 3)

# Visualize the result in a scatter plot
fviz_cluster(list(data = df, cluster = sub_grps))

# Plot the obtained dendrogram with
# rectangle borders for k clusters
plot(hc1, cex = 0.6, hang = -1)
rect.hclust(hc1, k = 3, border = 2:4)


col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(2,3))
fourfoldplot(cm_ct$table, color = col, conf.level = 0, margin = 1, main=paste("CTree (",round(cm_ct$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_nb$table, color = col, conf.level = 0, margin = 1, main=paste("NaiveBayes (",round(cm_nb$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_knn$table, color = col, conf.level = 0, margin = 1, main=paste("K-nn (",round(cm_ranger$overall[1]*100),"%)",sep=""))


cm_knn$table



