################################################## 

 #Telecom Churn using H2o
 #Load the data

##################################################


setwd("E:/Upgrad R/module 4/Neural network/Assignment")
install.packages("MASS")
install.packages("h2o")
install.packages("Hmisc")
install.packages("ggplot")
library(h2o)
library(Hmisc)
library(ggplot2)
h2o.init(nthreads=-1) ## use all avaliables cores in the machine 
churn <-read.csv("telecom_nn_train.csv", header = T)
View(churn)
str(churn)
describe(churn)

#Checking for NA

apply(churn,2,function(x) sum(is.na(x)))

#checking the churn wrt overrall expenses
# TotalCharges vs MonthlyCharges 
ggplot(churn, aes(x = TotalCharges, y = MonthlyCharges, col = tenure)) + geom_point()
#TotalCharges vs tenure 
ggplot(churn, aes(x = TotalCharges, y = tenure, col = tenure)) + geom_point()
# MonthlyCharges vs tenure
ggplot(churn, aes(x = MonthlyCharges, y = tenure, col = tenure)) + geom_point()
# churn vs non-churn customers
ggplot(churn, aes(x = churn$Churn)) + geom_bar()
#Converting target variable to factor
churn$Churn <- as.factor(churn$Churn)

# Bar Charts of attributes Vs target variable
# PhoneService Vs Churn
ggplot(churn, aes(x = churn$PhoneService, fill = Churn)) + geom_bar(position = "dodge")
# Dependents Vs Churn
ggplot(churn, aes(x = churn$Dependents, fill = factor(Churn))) + geom_bar(position = "dodge")
# Partner Vs Churn
ggplot(churn, aes(x = churn$Partner, fill = Churn)) + geom_bar(position = "dodge")
# gender Vs Churn
ggplot(churn, aes(x = factor(churn$gender), fill = Churn)) + geom_bar(position = "dodge")
# SeniorCitizen Vs Churn
ggplot(churn, aes(x = factor(churn$SeniorCitizen), fill = Churn)) + geom_bar(position = "dodge")

# Stacked bar charts can also be plotted
# PhoneService Vs Churn
ggplot(churn, aes(x = churn$PhoneService, fill = Churn)) + geom_bar(position = "stack")
# Dependents Vs Churn
ggplot(churn, aes(x = churn$Dependents, fill = factor(Churn))) + geom_bar(position = "stack")
# Partner Vs Churn
ggplot(churn, aes(x = churn$Partner, fill = Churn)) + geom_bar(position = "stack")
# gender Vs Churn
ggplot(churn, aes(x = factor(churn$gender), fill = Churn)) + geom_bar(position = "stack")
# SeniorCitizen Vs Churn
ggplot(churn, aes(x = factor(churn$SeniorCitizen), fill = Churn)) + geom_bar(position = "stack")

# Checking for outliers in numeric variables using box plots
boxplot(churn$tenure)
boxplot.stats(churn$tenure)$out
boxplot(churn$MonthlyCharges)
boxplot.stats(churn$MonthlyCharges)
boxplot(churn$TotalCharges)
boxplot.stats(churn$TotalCharges)
# No outliers

#model NN
churn_nn <- churn
#convert to numeric
churn_nn$PaperlessBilling <-as.numeric(churn_nn$PaperlessBilling)
churn_nn$Partner <- as.numeric(churn_nn$Partner)
churn_nn$Dependents<- as.numeric(churn_nn$Dependents)
churn_nn$PhoneService<- as.numeric(churn_nn$PhoneService)
churn_nn$gender<- as.numeric(churn_nn$gender)
str(churn_nn)
#No Need for dummy since we got a train data
#scaling the data

churn_nn$tenure <- scale(churn_nn$tenure)
churn_nn$MonthlyCharges <- scale(churn_nn$MonthlyCharges)
churn_nn$TotalCharges <- scale(churn_nn$TotalCharges)



#list of indicies for the data point - traning  & test data

index <- sample(1:nrow(churn_nn),round(0.75*nrow(churn_nn)))
train1 <- churn_nn[index,]
test1 <- churn_nn[-index,]

# Initialize the h2o environment
library(h2o)
h2o.init()
write.csv(train1, "trainchurn.csv", row.names = F)
write.csv(test1 , "testchurn.csv", row.names = F)

# Perform cross-validation on the training_frame

train1 <- h2o.importFile("trainchurn.csv")
test1 <- h2o.importFile("testchurn.csv")
nnet <- h2o.deeplearning(names(train1[, -14]), names(train1[, 14]), train1, distribution = "gaussian", activation = "TanhWithDropout", hidden = c(100, 100, 100), hidden_dropout_ratios = c(0.1, 0.1, 0.1), epochs = 10)
nnet
summary(nnet)
# Test the model on the test data

prediction <- h2o.predict(nnet, train1[,-14])
mse_train<-sum((train1[,14]-prediction)^2)/nrow(train1[,1])
prediction_1 <- h2o.predict(nnet, test1[,-14])
mse_test<-sum((test1[,14]-prediction_1)^2)/nrow(test1[,1])
mse_train
mse_test
write.table(as.matrix(prediction_1), file="telecom_nn_test.csv", row.names=FALSE, sep=",")

#Checkpoint 1: Neural Networks - Tuning hyperparameters WITHOUT epochs
#no passes in the traning set 
#we are not specifing how many times the data to iterate 
# small network, runs faster & more hidden layers -> more complex interactions
nnet_1 <- h2o.deeplearning(names(telecom_train[, -10]), 
                           names(telecom_train[, 10]),
                           training_frame = telecom_train, 
                           validation_frame = telecom_validation,
                           distribution = "bernoulli", 
                           activation = "RectifierWithDropout", 
                           hidden = c(300,300,300), 
                           hidden_dropout_ratios = c(0.1, 0.1, 0.1), 
                           epochs = 1,
                           standardize = TRUE) 

nnet_1
summary(nnet_1)
plot(nnet_1) #high error rate

nnet_2 <- h2o.deeplearning(names(telecom_train[, -10]), 
                           names(telecom_train[, 10]),
                           training_frame = telecom_train, 
                           validation_frame = telecom_validation,
                           distribution = "bernoulli", 
                           activation = "RectifierWithDropout", 
                           hidden = c(300,300,300, 300), 
                           hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                           epochs = 1,
                           standardize = TRUE) 
nnet_2 #slight improvment
nnet_3 <- h2o.deeplearning(names(telecom_train[, -10]), 
                           names(telecom_train[, 10]),
                           training_frame = telecom_train, 
                           validation_frame = telecom_validation,
                           distribution = "bernoulli", 
                           activation = "TanhWithDropout", 
                           hidden = c(300,300,300, 300), 
                           hidden_dropout_ratios = c(0.1, 0.1, 0.1, 0.1), 
                           epochs = 1,
                           standardize = TRUE) 

nnet_3



#Checkpoint 2: Neural Networks - Tuning hyperparameters WITH epochs 

nnet1 <- h2o.deeplearning(names(train1[, -10]),   #column number of predictors
                          names(train1[, 10]), 
                          train1,  # data in H2O format
                          distribution = "bernoulli", 
                          activation = "RectifierWithDropout",   
                          hidden = c(300, 300, 300),# three layers of 50 nodes
                          hidden_dropout_ratios = c(0.1, 0.1, 0.1,0.1),   # % for nodes dropout
                          epochs = 6,
                          standardize = TRUE) # max. no. of epochs
nnet1
summary(nnet1)     #we can see futhur improvment of R square metric which suggest that the model had captured large chunk of unexplained variables in the model
plot(nnet1)  #overfitting  occcurs because of large numbers of layers withmany neurons   
#by using sigle thinned network with smaller weights we can reduce over fitting
nnet1 <- h2o.deeplearning(names(train1[, -10]),   #column number of predictors
                          names(train1[, 10]), 
                          train1,  # data in H2O format
                          distribution = "bernoulli", 
                          activation = "RectifierWithDropout",   
                          hidden = c(300, 300, 300),# three layers of 50 nodes
                          hidden_dropout_ratios = c(0.1, 0.1, 0.1,0.1),   # % for nodes dropout
                          epochs = 20,
                          standardize = TRUE) # max. no. of epochs
nnet1
#Checkpoint 3

#Using model for Prediction

prediction2 <- h2o.predict(nnet1, train1[,-10])
Churn_train<-sum((train1[,10]-prediction2)^2)/nrow(train1[,1])
prediction_Churn <- h2o.predict(nnet1, test1[,-10])
Churn_test<-sum((test1[,10]-prediction_Churn)^2)/nrow(test1[,1])
Churn_train
Churn_test
write.table(as.matrix(prediction_Churn), file="telecom_nn_test.csv", row.names=FALSE, sep=",")
#Accuracy is 82% Specifity is 78% Sensitivty is 76%

# The  model with epoch got the best Accuarcy 
#this model got the max precision ie TP/Tp+Fp  & we got max specificty ie negatives that are correctly identified
#Overfitting problem of the model is redcued by optimizing the hyperparameters



