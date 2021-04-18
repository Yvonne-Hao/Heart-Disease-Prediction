setwd("/Users/Yvonne.Hao/Desktop/JHU/Spring Session 1/Data Analytics/project")
library(caret)
heart=read.csv("cleveland.data",na.strings = "?",sep=",",header=FALSE)
summary(heart)
sum(is.na(heart))

names(heart)=c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                "thalach","exang", "oldpeak","slope", "ca", "thal", "num")
heart=na.omit(heart)

heart$num[which(heart$num>0)]=1
#or heart$num[heart$num > 0] <- 1
sum(heart$num==1)
sum(heart$num==0)
#heart$num<-factor(heart$num)
str(heart)
summary(heart)

boxplot(heart, main="boxplot")

#Results of having disease or not. 
barplot(table(heart$num),
        main="Distribution of num", col="#d33b3b")

feature.names=names(heart)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
convert.magic <- function(obj,types){
  out <- lapply(1:length(obj),FUN = function(i){FUN1 <- 
    switch(types[i],
           character = as.character,
           numeric = as.numeric,
           factor = as.factor); FUN1(obj[,i])})
  names(out) <- colnames(obj)
  as.data.frame(out)
}

heart <- convert.magic(heart,chclass)
heartlabel<-heart ##used for lable only
levels(heartlabel$num) = c("No disease","Disease")
levels(heartlabel$sex) = c("female","male")

mosaicplot(~ num+sex+cp, data = heartlabel,
           main="Fate by Gender and Chest Pain Type", shade=FALSE,color=TRUE,
           xlab="Heart disease", ylab="Gender")
mosaicplot(heartlabel$cp ~ heartlabel$num,
           main="Fate by Chest Pain Type", shade=FALSE,color=TRUE,
           xlab="Chest Pain Type", ylab="Heart disease")
mosaicplot(heartlabel$fbs ~ heartlabel$num,
           main="Fate by Fasting Blood Sugar", shade=FALSE,color=TRUE,
           xlab="Fasting Blood Sugar", ylab="Heart disease")
mosaicplot(heartlabel$restecg ~ heartlabel$num,
           main="Fate by Restecg", shade=FALSE,color=TRUE,
           xlab="Restecg", ylab="Heart disease")
mosaicplot(heartlabel$exang ~ heartlabel$num,
           main="Fate by Exercise Induced Angina", shade=FALSE,color=TRUE,
           xlab="Exercise Induced Angina", ylab="Heart disease")
mosaicplot(heartlabel$slope ~ heartlabel$num,
           main="Fate by Slope", shade=FALSE,color=TRUE,
           xlab="Slope", ylab="Heart disease")
mosaicplot(heartlabel$ca ~ heartlabel$num,
           main="Fate by Number of Major Vessels", shade=FALSE,color=TRUE,
           xlab="Number of Major Vessels", ylab="Heart disease")
mosaicplot(heartlabel$thal ~ heartlabel$num,
           main="Fate by Thal", shade=FALSE,color=TRUE,
           xlab="Number of Thal", ylab="Heart disease")
boxplot(heartlabel$age ~ heartlabel$num,
        main="Fate by Age",
        ylab="Age",xlab="Heart disease")
boxplot(heartlabel$trestbps ~ heartlabel$num,
        main="Fate by Resting Blood Pressure",
        ylab="Resting Blood Pressure",xlab="Heart disease")
boxplot(heartlabel$chol ~ heartlabel$num,
        main="Fate by Serum Cholesterol",
        ylab="Serum Cholesterol",xlab="Heart disease")
boxplot(heartlabel$thalach ~ heartlabel$num,
        main="Fate by Maximum Heart Rate Achieved",
        ylab="Maximum Heart Rate Achieved",xlab="Heart disease")
boxplot(heartlabel$oldpeak ~ heartlabel$num,
        main="Fate by Oldpeak",
        ylab="Oldpeak",xlab="Heart disease")



set.seed(6)
train=sample(1:nrow(heart),7*nrow(heart)/10)
##(1:nrow) is a range
heart.train=heart[train,]  ##[row, column]
heart.test=heart[-train,]
dim(heart.train)
dim(heart.test)


heart.train.output=heart.train[,-which(colnames(heart)=="num")]  ##excluding/minus column "num" as it is the response variable
heart.test.output=heart.test[,-which(colnames(heart)=="num")]

heart.train.input=heart.train$num
heart.test.input=heart.test$num

#Decision tree model
library(tree)
attach(heart)
RNGkind(sample.kind = "Rounding")
set.seed(5)
tree.heart = tree(num ~ ., heart, subset=train)
tree.pred = predict(tree.heart,heart.test,type="class")

#confusion matrix
table(tree.pred, heart.test.input)
# calculate prediction accuracy
mean(tree.pred==heart.test.input)
summary(tree.heart)
#plot the tree
plot(tree.heart)
text(tree.heart,pretty=0)

# use cross-validation and pruning to obtain smaller trees and their prediction accuracy 
cv.heart = cv.tree(tree.heart,FUN=prune.misclass)
names(cv.heart)
#When size is 3 has the minimum div
cv.heart
# use a function called prune.misclass() to obtain the best tree, which we know has a size of 3
prune.heart = prune.misclass(tree.heart,best=3)
# plot the best tree
plot(prune.heart)
# display branch names on the tree
text(prune.heart, pretty = 0)
# using the best tree to predict the test data
tree.pred2 = predict(prune.heart,heart.test,type="class")
# display a table that shows predictions for test data versus actuals for test data
confusion<-table(tree.pred2, heart.test.input)
# calculate prediction accuracy
mean(tree.pred2==heart.test.input)
#calculate F1
tree.F1<-2*confusion[1,1]/(2*confusion[1,1]+confusion[2,1]+confusion[1,2])
tree.F1

#Random forest
library(pROC)
library(randomForest)
RNGkind(sample.kind = "Rounding")
set.seed(4)
RF.heart <- randomForest(num ~  age+sex+cp+
                                      trestbps+chol+fbs+
                                      restecg+thalach+exang+oldpeak+slope+ca+thal,
                                    data = heart.train,
                                    ntree =500,
                                    mtry=4,
                                    importance=TRUE ,
                                    proximity=TRUE)
plot(RF.heart)
RF.heart$importance
varImpPlot(RF.heart, main = "Variable Importance")
#predict test data using RF model
RF.pred <- predict(RF.heart,newdata=heart.test)
#confusion matrix
RF.confusion<-table(RF.pred, heart.test.input)
#prediction accuracy
mean(RF.pred==heart.test.input)
#ROC plot
ran_roc <- roc(heart.test.y,as.numeric(RF.pred))
plot(ran_roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("black", "black"), max.auc.polygon=TRUE,auc.polygon.col="lightyellow", print.thres=TRUE,main='ROC plot of RF model,mtry=4,ntree=500')
RF.F1<-2*RF.confusion[1,1]/(2*RF.confusion[1,1]+RF.confusion[2,1]+RF.confusion[1,2])
RF.F1

#SVM
for (f in feature.names) {
  if (class(heart[[f]])=="factor") {
    levels <- unique(c(heart[[f]]))
    heart[[f]] <- factor(heart[[f]],
                         labels=make.names(levels))
  }
}
set.seed(6)
train=sample(1:nrow(heart),7*nrow(heart)/10)
heart.train=heart[train,]
heart.test=heart[-train,]
dim(heart.train)
dim(heart.test)


heart.train.x=heart.train[,-which(colnames(heart)=="num")]
heart.test.x=heart.test[,-which(colnames(heart)=="num")]

heart.train.y=heart.train$num
heart.test.y=heart.test$num

set.seed(10)
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary)
SVM.heart <- train(num ~ ., data = heart.train,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
SVM.pred <- predict(SVM.heart, heart.test)
SVM.confusion<-table(SVM.pred, heart.test.y)
mean(SVM.pred ==heart.test.y)
SVM.F1<-2*SVM.confusion[1,1]/(2*SVM.confusion[1,1]+SVM.confusion[2,1]+SVM.confusion[1,2])
SVM.F1
#ROC Curve
ROC.svm <- roc(as.numeric(heart.test.y),as.numeric(SVM.pred))
plot(ROC.svm, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("black", "black"), max.auc.polygon=TRUE,auc.polygon.col="lightyellow", print.thres=TRUE,main='ROC plot of SVM model')

#K-means
RNGkind(sample.kind = "Rounding")
set.seed(9)

# run K-means using x as our data, to create 2 clusters, and with 50 different initializations
km.out=kmeans(heart.train.x,2,nstart=50)
plot(heart.train.x,col=km.out$cluster,pch=km.out$cluster,lwd=2)
plot(heart$sex, heart$cp, col=km.out$cluster,pch=km.out$cluster,xlab="sex", ylab="cp") 

# display the results
km.out
km.pred <- predict(km.out, heart.test.x)
# plot the results to see clusters
plot(heart.train.x,col=km.out2$cluster,pch=km.out$cluster)