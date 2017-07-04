#load libraries
library(plyr)
library(ggplot2)
library(MASS)
library(dplyr)
library(Metrics)
library(gbm)
library(cvAUC)
library(rpart)
library(e1071)
library(rpart.plot)
library(caret)
library(mlbench)
library(dummies)
library(vegan)
library(lars)
library(glmnet)
library(xgboost)
library(missForest)
library(clue)
library(plotmo) 

rm(list = ls())
#LoadDatasets
setwd("~/Desktop/BigMartSales")
train <-read.csv("Train_UWu5bXk.csv")
test <-read.csv("Test_u94Q5KV.csv")
dim(train)
dim(test)
summary(train)
#Items with visibility <0.2 contribute to bulk of sales
ggplot(train, aes(x= Item_Visibility, y = Item_Outlet_Sales)) + geom_point(size = 2.0, color="blue") + xlab("Item Visibility") + ylab("Item Outlet Sales") +ggtitle("Item Visibility vs Item Outlet Sales")

#Outlet OUT027 appears to be a flagship
ggplot(train, aes(Outlet_Identifier, Item_Outlet_Sales)) + geom_bar(stat ="identity", color = "blue") +theme(axis.text.x = element_text(angle = 70, vjust =0.5, color = "black")) + ggtitle("Outlets vs Total Sales") + theme_bw()

#Fruit and veg contribute more to the sales followed by Snack Foods
ggplot(train, aes(Item_Type, Item_Outlet_Sales)) + geom_bar( stat = "identity",color = "blue")+theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "navy")) +xlab("Item Type") + ylab("Item Outlet Sales")+ggtitle("Item Type vs Sales")

#Health and Hygiene Item Type has outlier, more MRP spread in dairy
ggplot(train, aes(Item_Type, Item_MRP)) +geom_boxplot() +ggtitle("Box Plot") + theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "red")) + xlab("Item Type") + ylab("Item MRP") + ggtitle("Item Type vs Item MRP")

#Seafood has more spread within 5000, others Item_Types are fluctuating
ggplot(train, aes(Item_Type, Item_Outlet_Sales)) +geom_boxplot() +ggtitle("Box Plot") + theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "red")) + xlab("Item Type") + ylab("Item_Outlet_Sales") + ggtitle("Item Type vs Item_Outlet_Sales")

#Outlet_Establishmnet Year Vs OutletSales
ggplot(train, aes(Outlet_Establishment_Year, Item_Outlet_Sales)) + geom_bar(stat ="identity", color = "blue") +theme(axis.text.x = element_text(angle = 70, vjust =0.5, color = "black")) + ggtitle("Established Year vs Total Sales") + theme_bw()


#create_new_column and assign a value
test$Item_Outlet_Sales <- 1

#combine train and test data
combi <- rbind(train, test)

#rename level in Outlet_Size
levels(combi$Outlet_Size)[1] <- "Other"

#standardise levels of Item_Fat_Content

combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content,c("LF" = "Low Fat", "reg" ="Regular"))
combi$Item_Fat_Content <- revalue(combi$Item_Fat_Content, c("low fat" = "Low Fat"))
#Item_Weight has 1463 missing values in train

prop.table(table(is.na(train$Item_Weight)))
#almost 17% values are missing->removing the column

# established years until 2013
combi$Outlet_Establishment_Year <-(2013-combi$Outlet_Establishment_Year)

# #Cluster based on Item Type
# set.seed(1234)
# salescluster<-kmeans(combi[5],3,nstart = 20)
# salescluster$centers
# plot(train,col=salescluster$cluster)

#drop insignificant variables not required in modeling
#combi <- dplyr::select(combi, -c(Item_Identifier,Item_Type,Item_Weight,Outlet_Identifier))
combi <- dplyr::select(combi, -c(Item_Identifier,Item_Weight,Outlet_Identifier))


#train and test sets after imputing 
#divide data set
set.seed(1234)
Rtrain <- combi[1:nrow(train),]
Rtest <- combi[-(1:nrow(train)),]

#get categorical variables in a df
data_catg<-combi[,c("Item_Fat_Content","Outlet_Size","Outlet_Location_Type","Outlet_Type")]

data_catg_dummy <- dummy.data.frame(data_catg,sep = "_")
str(data_catg_dummy)
data_num<-combi[,c('Item_Visibility','Item_MRP','Outlet_Establishment_Year')]
data_num <- data.frame(apply(data_num,2,function(x){as.character(x)}))
data_num <- data.frame(apply(data_num,2,function(x){as.numeric(x)}))

#standardise numerical data using range method
independent_Variables = decostand(data_num, "range")

# get target variable in a df
target <-subset(combi,select = ("Item_Outlet_Sales"))

Final_Data <-data.frame(data_catg_dummy,independent_Variables,target)
str(Final_Data)

#divide data set
set.seed(1234)
new_train <- Final_Data[1:nrow(train),]
new_test <- Final_Data[-(1:nrow(train)),]


#NNetModel
mygrid <- expand.grid(.decay=seq(0.0009,0.003,0.0001), .size=c(1:3))

#mygrid <- expand.grid(.decay=.00162, .size=c(1:3))

#check range of sales
summary(new_train$Item_Outlet_Sales)
maxSales=max(new_train$Item_Outlet_Sales)
#normalize to get a value between 0 and 1 divide by max sales to keep in sync with other standardised numerical vars
nnetfit <- train(Item_Outlet_Sales/13086.9648~., data = new_train, method="nnet", maxit=1000, tuneGrid=mygrid, trace=F)
print(nnetfit)
plot(nnetfit)

#error metrics train
nnetfit.predict<-predict(nnetfit,new_train)
nnetfit.predict<-nnetfit.predict*maxSales#to offset division by max sales value
rmse(new_train$Item_Outlet_Sales, nnetfit.predict)#1070.062#1071.091#1069.494#1068.908#1068.904#1069.31#1068.767#1068.736#1068.743

#error metrics test
nnetfit.predict.test<-predict(nnetfit,new_test)
nnetfit.predict.test<-nnetfit.predict.test*maxSales#to offset division by max sales value
rmse(new_test$Item_Outlet_Sales, nnetfit.predict.test)
#2554.366#2552.303#2550.059#2555.213#2555.412#2555.373#2555.315#2555.703#2555.371#2555.367


# NNET Data for submission 
submissionFile <- data.frame(Item_Identifier = test$Item_Identifier, Outlet_Identifier = test$Outlet_Identifier,Item_Outlet_Sales =nnetfit.predict.test)
write.csv(submissionFile,"Submission_NNet.csv",row.names = FALSE)

