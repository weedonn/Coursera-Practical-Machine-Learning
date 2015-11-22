setwd("C:\\Users\\Nigel\\Documents\\GitHub\\Coursera-Practical-Machine-Learning")
# Type ls() to see a list of the variables in your workspace. Then, type
# rm(list=ls()) to clear your workspace.
rm(list=ls())
library("randomForest")   # Used by caret
#library("corrplot")       # plot correlations
#library("doParallel")     # parallel processing
#library("dplyr")          # Used by caret
#library("gbm")            # Boosting algorithms
#library("kernlab")        # support vector machine
#library("partykit")       # Plotting trees
#library("pROC")           # plot the ROC curve
#library("rpart")          # CART algorithm for decision trees
library("caret")
library("rattle")         # Data and model analysis workbench
#rattle()

# Set the random seed.
set.seed(1618)
# Function to generate files with predictions to submit for assignment
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

# Function to read files with predictions into character vector
pml_read_files = function(x){
        fileVector <- vector()
        fileContents <- vector()
        for(i in 1:x){
                filename = paste0("problem_id_",i,".txt")
                fileContents <- read.table(filename)
                fileVector <- c(fileVector, paste0(fileContents[[1]]))
        }
        fileVector
}

#submittedAnswers <- pml_read_files(20)

# Set up training control
# trainCtrl <- trainControl(method="repeatedcv",          # 10 fold cross validation
#                      repeats=5,                         # do 5 repititions of cv
#                      classProbs=TRUE)
trainCtrl <- trainControl(method="cv",          # 10 fold cross validation
                          classProbs=TRUE)


# Set up to to do parallel processing
# registerDoParallel(4)       # Registrer a parallel backend for train
# getDoParWorkers()

# Load the Training and Testing datasest into data frames.
#pml_training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
pml_training_Url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
pml_training <- read.csv(url(pml_training_Url), na.strings=c("NA","#DIV/0!",""))
# Remove the first 7 columns which contain Identity and Categorical features, which are not predictive.
pml_training <- pml_training[, -c(1:7)]
# Discard the columns where there is < 95% observations
# sum(colSums(!is.na(pml_training)) < nrow(pml_training) * 0.95) #columns with more than 95% data missing
cols2Discard <- c(colSums(!is.na(pml_training)) >= nrow(pml_training) * 0.95)
pml_training   <-  pml_training[,cols2Discard]

# Identify "near-zero-variance" predictors.
# These predictors may have an undue influence on the model and should be eliminated prior to modeling.
#nzv2Remove <- nearZeroVar(pml_training, saveMetrics= TRUE)
#nzv2Remove
#pml_training <- pml_training[, nzv2Remove$nzv==FALSE]

#colnames(pml_training)

#pml_testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
pml_testing_Url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
pml_testing <- read.csv(url(pml_testing_Url), na.strings=c("NA","#DIV/0!",""))
# Remove the first 7 columns which contain Identity and Categorical features, which are not predictive.
pml_testing <- pml_testing[ , -c(1:7)]
# Discard the columns where there is < 95% observations
pml_testing <- pml_testing[,cols2Discard]
# Identify and remove "near-zero-variance" predictors.
#pml_testing <- pml_testing[, nzv2Remove$nzv==FALSE]

#colnames(pml_training)
#colnames(pml_testing)

# Partioning pml_training data set into two data sets, 70% for training, 30% for validation(testing):
inTrain <- createDataPartition(pml_training$classe, p=0.7, list=FALSE)
training <- pml_training[inTrain, ]
testing <- pml_training[-inTrain, ]

# Partioning the training data set again into a 20% sample data set to estimate the OOB of the final model.
inSample <- createDataPartition(training$classe, p=0.2, list=FALSE)
OOBtraining <- training[inSample, ]
#rfOOBModel <- randomForest(OOBtraining$classe ~.,data=OOBtraining, importance = TRUE)
rfOOBModel <- train(OOBtraining$classe ~ ., method="rf", trControl=trainControl(method="cv"), data=OOBtraining, importance = TRUE)
rfOOBModel
rfPredict <- predict(rfOOBModel,testing)
rfConfMatrix <- confusionMatrix(rfPredict, testing$classe)
rfConfMatrix

#rfOOBModelFM <- rfOOBModel$finalModel
#varImp(rfOOBModelFM, scale=FALSE)

# Variable importance analysis
rfImportance <- varImp(rfOOBModel,scale=FALSE) # for train(method="rf")
ImpMeasure<-data.frame(rfImportance$importance)
ImpMeasure$Vars<-row.names(ImpMeasure)
ImpMeasure[order(-ImpMeasure$Overall),][1:15,]

plot(rfImportance, top = 15) #dim(rfImportance$importance)[1])
#varImpPlot(rfOOBModel) # for randomForestModel()

#importance(rfOOBModelFM)

randomForestModel <- train(training$classe ~ ., method="rf", trControl=trainControl(method="cv"), data=training, importance = TRUE)
#randomForestModel <- randomForest(training$classe ~.,data=training, importance = TRUE)
randomForestModel
randomForestPredict <- predict(randomForestModel,testing)
randomForestConfMatrix <- confusionMatrix(randomForestPredict, testing$classe)
randomForestConfMatrix
predictors(randomForestModel)


trainingPredictors <- training[,1:ncol(training)-1]
testingPredictors <- testing[,1:ncol(testing)-1]

# K-Nearest Neighbour Analysis(knn)
# knn requires variables to be normalized or scaled.
KNNctrl = trainControl(method="cv")
KNNmod = train(training$classe ~ ., data=training, method = "knn", trControl = KNNctrl, preProcess = c("center","scale"))
# Output of knn trained model
KNNmod
predictors(KNNmod)

#Plotting yields Number of Neighbours Vs accuracy (based on cross validation)
plot(KNNmod)

KNNpred <- predict(KNNmod,testing)
KNNConfMatrix <- confusionMatrix(KNNpred, testing$classe)
KNNConfMatrix

# Generate the predicted answers
predictedAnswers <- predict(randomForestModel,newdata=pml_testing)
#predictedAnswers <- predict(KNNmod,newdata=pml_testing)
saveRDS(predictedAnswers, file="predictedAnswers.rds")
#rm(predictedAnswers)
#predictedAnswers <- readRDS("predictedAnswers.rds")
#write.table(t(predictedAnswers), file="predictedAnswers.txt",row.names=F, col.names=F)  # t() to give row of data.
#write.table(t(predictedAnswers), file="predictedAnswers.txt",row.names=F, col.names=F, append=TRUE)
#predictedAnswers

# Generate the the 20 prediction files to submit
#pml_write_files(predictedAnswers)
