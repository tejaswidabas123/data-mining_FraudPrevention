# Load required libraries
library(caret)
library(e1071)
library(randomForest)
library(kknn)
library(rpart)
library(naivebayes)
library(nnet)
library(gbm)

# Load data
data <- read.csv("consumer_interactions.csv", stringsAsFactors = FALSE)
any(is.na(data$Region))
any(is.na(data$State))
any(is.na(data$Area))
any(is.na(data$City))
any(is.na(data$Consumer_profile))
any(is.na(data$Product_category))
any(is.na(data$Product_type))
any(is.na(data$AC_1001_Issue))
any(is.na(data$AC_1002_Issue))
any(is.na(data$AC_1003_Issue))
any(is.na(data$TV_2001_Issue))
any(is.na(data$TV_2002_Issue))
any(is.na(data$TV_2003_Issue))
any(is.na(data$Claim_Value))
any(is.na(data$Service_Centre))
any(is.na(data$Product_Age))
any(is.na(data$Purchased_from))
any(is.na(data$Call_details))
any(is.na(data$Purpose))
any(is.na(data$Fraud))

mean_value <- mean(data$Claim_Value, na.rm = TRUE)
data[is.na(data$Claim_Value), "Claim_Value"] <- mean_value

any(is.na(data$Claim_Value))


data <- data[,-1]

#Plot
DF1 <- data[, c("Claim_Value","Service_Centre","Product_Age")]
pairs(DF1)
DF2 <- data[, c("Claim_Value","Service_Centre","Product_Age","Call_details")]
pairs(DF2)
library(corrplot)
cor_mat <- cor(DF2)
corrplot(cor_mat, method="color", type="upper", tl.col="black", tl.srt=45)

# Convert target variable 'Fraud' to factor
data$Fraud <- as.factor(data$Fraud)

# Set seed for reproducibility
set.seed(123)

# Split dataset into training and test datasets
trainIndex <- createDataPartition(data$Fraud, p = 0.66, list = FALSE)
train <- data[trainIndex,]
test <- data[-trainIndex,]

####################################################################################################
# Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model <- train(Fraud ~ ., data = train, method = "glm", trControl = control)
svm_model <- train(Fraud ~ ., data = train, method = "svmLinear", trControl = control)
rf_model <- train(Fraud ~ ., data = train, method = "rf", trControl = control)
knn_model <- train(Fraud ~ ., data = train, method = "kknn", trControl = control)
cart_model <- train(Fraud ~ ., data = train, method = "rpart", trControl = control)
nnet_model <- train(Fraud ~ ., data = train, method = "nnet", trControl = control)
gbm_model <- train(Fraud ~ ., data = train, method = "gbm", trControl = control)
weka_model <- train(Fraud ~ ., data=train, method="J48", trControl=control)


# Make predictions on the test set and evaluate the models
glm_predictions <- predict(glm_model, newdata = test)
glm_cm <- confusionMatrix(glm_predictions, test$Fraud)
glm_cm

svm_predictions <- predict(svm_model, newdata = test)
svm_cm <- confusionMatrix(svm_predictions, test$Fraud)
svm_cm

rf_predictions <- predict(rf_model, newdata = test)
rf_cm <- confusionMatrix(rf_predictions, test$Fraud)
rf_cm

knn_predictions <- predict(knn_model, newdata = test)
knn_cm <- confusionMatrix(knn_predictions, test$Fraud)
knn_cm

cart_predictions <- predict(cart_model, newdata = test)
cart_cm <- confusionMatrix(cart_predictions, test$Fraud)
cart_cm

nnet_predictions <- predict(nnet_model, newdata = test)
nnet_cm <- confusionMatrix(nnet_predictions, test$Fraud)
nnet_cm

gbm_predictions <- predict(gbm_model, newdata = test)
gbm_cm <- confusionMatrix(gbm_predictions, test$Fraud)
gbm_cm

weka_predictions <- predict(weka_model, newdata = test)
weka_cm <- confusionMatrix(weka_predictions, test$Fraud)
weka_cm

##############################################################################################

tp_rate <- glm_cm$byClass[2]
fp_rate <- glm_cm$byClass[3]

precision <- glm_cm$byClass[1]
recall <- glm_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(glm_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm$byClass[2]
fp_rate <- svm_cm$byClass[3]

precision <- svm_cm$byClass[1]
recall <- svm_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(svm_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm$byClass[2]
fp_rate <- rf_cm$byClass[3]

precision <- rf_cm$byClass[1]
recall <- rf_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(rf_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm$byClass[2]
fp_rate <- knn_cm$byClass[3]

precision <- knn_cm$byClass[1]
recall <- knn_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(knn_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm$byClass[2]
fp_rate <- cart_cm$byClass[3]

precision <- cart_cm$byClass[1]
recall <- cart_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(cart_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm$byClass[2]
fp_rate <- nnet_cm$byClass[3]

precision <- nnet_cm$byClass[1]
recall <- nnet_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm$byClass[2]
fp_rate <- gbm_cm$byClass[3]

precision <- gbm_cm$byClass[1]
recall <- gbm_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(gbm_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm$byClass[2]
fp_rate <- weka_cm$byClass[3]

precision <- weka_cm$byClass[1]
recall <- weka_cm$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Recursive feature elimination 
attrSel <- rfe(x = train[, -which(names(train) %in% c("Fraud"))], y = train$Fraud, sizes = c(1:8), rfeControl = rfeControl(functions = rfFuncs))
rfe(x = train[, -which(names(train) %in% c("Fraud"))], y = train$Fraud, sizes = c(1:8), rfeControl = rfeControl(functions = rfFuncs))
reducedTrain_1 <- train[, c("Fraud", attrSel$optVariables)]
reducedTest_1 <- test[, c("Fraud", attrSel$optVariables)]

# Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "glm", trControl = control)
svm_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "svmLinear", trControl = control)
rf_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "rf", trControl = control)
knn_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "kknn", trControl = control)
cart_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "rpart", trControl = control)
nnet_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "nnet", trControl = control)
gbm_model1 <- train(Fraud ~ ., data = reducedTrain_1, method = "gbm", trControl = control)
weka_model1 <- train(Fraud ~ ., data=reducedTrain_1, method="J48", trControl=control)


# Make predictions on the test set and evaluate the models
glm_predictions1 <- predict(glm_model1, newdata = reducedTest_1)
glm_cm1 <- confusionMatrix(glm_predictions1, reducedTest_1$Fraud)
glm_cm1

svm_predictions1 <- predict(svm_model1, newdata = reducedTest_1)
svm_cm1 <- confusionMatrix(svm_predictions1, reducedTest_1$Fraud)
svm_cm1

rf_predictions1 <- predict(rf_model1, newdata = reducedTest_1)
rf_cm1 <- confusionMatrix(rf_predictions1, reducedTest_1$Fraud)
rf_cm1

knn_predictions1 <- predict(knn_model1, newdata = reducedTest_1)
knn_cm1 <- confusionMatrix(knn_predictions1, reducedTest_1$Fraud)
knn_cm1

cart_predictions1 <- predict(cart_model1, newdata = reducedTest_1)
cart_cm1 <- confusionMatrix(cart_predictions1, reducedTest_1$Fraud)
cart_cm1

nnet_predictions1 <- predict(nnet_model1, newdata = reducedTest_1)
nnet_cm1 <- confusionMatrix(nnet_predictions1, reducedTest_1$Fraud)
nnet_cm1

gbm_predictions1 <- predict(gbm_model1, newdata = reducedTest_1)
gbm_cm1 <- confusionMatrix(gbm_predictions1, reducedTest_1$Fraud)
gbm_cm1

weka_predictions <- predict(weka_model1, newdata = reducedTest_1)
weka_cm1 <- confusionMatrix(weka_predictions, reducedTest_1$Fraud)
weka_cm1



##############################################################################################

tp_rate <- glm_cm1$byClass[2]
fp_rate <- glm_cm1$byClass[3]

precision <- glm_cm1$byClass[1]
recall <- glm_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(glm_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm1$byClass[2]
fp_rate <- svm_cm1$byClass[3]

precision <- svm_cm1$byClass[1]
recall <- svm_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(svm_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm1$byClass[2]
fp_rate <- rf_cm1$byClass[3]

precision <- rf_cm1$byClass[1]
recall <- rf_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(rf_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm1$byClass[2]
fp_rate <- knn_cm1$byClass[3]

precision <- knn_cm1$byClass[1]
recall <- knn_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(knn_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm1$byClass[2]
fp_rate <- cart_cm1$byClass[3]

precision <- cart_cm1$byClass[1]
recall <- cart_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(cart_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm1$byClass[2]
fp_rate <- nnet_cm1$byClass[3]

precision <- nnet_cm1$byClass[1]
recall <- nnet_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(nnet_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm1$byClass[2]
fp_rate <- gbm_cm1$byClass[3]

precision <- gbm_cm1$byClass[1]
recall <- gbm_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_1$Fraud, as.numeric(gbm_predictions1))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions1), as.numeric(reducedTest_1$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm1$byClass[2]
fp_rate <- weka_cm1$byClass[3]

precision <- weka_cm1$byClass[1]
recall <- weka_cm1$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Recursive feature elimination with cross-validation (RFECV)
rfecv_fit <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
rfecv_res <- rfe(x = train[, -which(names(train) %in% c("Fraud"))], y = train$Fraud, sizes = c(1:8), rfeControl = rfecv_fit)
reducedTrain_2 <- train[, c("Fraud", rfecv_res$optVariables)]
reducedTest_2 <- test[, c("Fraud", rfecv_res$optVariables)]

# Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "glm", trControl = control)
svm_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "svmLinear", trControl = control)
rf_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "rf", trControl = control)
knn_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "kknn", trControl = control)
cart_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "rpart", trControl = control)
nnet_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "nnet", trControl = control)
gbm_model2 <- train(Fraud ~ ., data = reducedTrain_2, method = "gbm", trControl = control)
weka_model2 <- train(Fraud ~ ., data=reducedTrain_2, method="J48", trControl=control)


# Make predictions on the test set and evaluate the models
glm_predictions2 <- predict(glm_model2, newdata = reducedTest_2)
glm_cm2 <- confusionMatrix(glm_predictions2, reducedTest_2$Fraud)
glm_cm2

svm_predictions2 <- predict(svm_model2, newdata = reducedTest_2)
svm_cm2 <- confusionMatrix(svm_predictions2, reducedTest_2$Fraud)
svm_cm2

rf_predictions2 <- predict(rf_model2, newdata = reducedTest_2)
rf_cm2 <- confusionMatrix(rf_predictions2, reducedTest_2$Fraud)
rf_cm2

knn_predictions2 <- predict(knn_model2, newdata = reducedTest_2)
knn_cm2 <- confusionMatrix(knn_predictions2, reducedTest_2$Fraud)
knn_cm2

cart_predictions2 <- predict(cart_model2, newdata = reducedTest_2)
cart_cm2 <- confusionMatrix(cart_predictions2, reducedTest_2$Fraud)
cart_cm2

nnet_predictions2 <- predict(nnet_model2, newdata = reducedTest_2)
nnet_cm2 <- confusionMatrix(nnet_predictions2, reducedTest_2$Fraud)
nnet_cm2

gbm_predictions2 <- predict(gbm_model2, newdata = reducedTest_2)
gbm_cm2 <- confusionMatrix(gbm_predictions2, reducedTest_2$Fraud)
gbm_cm2

weka_predictions2 <- predict(weka_model2, newdata = reducedTest_2)
weka_cm2 <- confusionMatrix(weka_predictions2, reducedTest_2$Fraud)
weka_cm2

##############################################################################################

tp_rate <- glm_cm2$byClass[2]
fp_rate <- glm_cm2$byClass[3]

precision <- glm_cm2$byClass[1]
recall <- glm_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(glm_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm2$byClass[2]
fp_rate <- svm_cm2$byClass[3]

precision <- svm_cm2$byClass[1]
recall <- svm_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(svm_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm2$byClass[2]
fp_rate <- rf_cm2$byClass[3]

precision <- rf_cm2$byClass[1]
recall <- rf_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(rf_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm2$byClass[2]
fp_rate <- knn_cm2$byClass[3]

precision <- knn_cm2$byClass[1]
recall <- knn_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(knn_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm2$byClass[2]
fp_rate <- cart_cm2$byClass[3]

precision <- cart_cm2$byClass[1]
recall <- cart_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(cart_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm2$byClass[2]
fp_rate <- nnet_cm2$byClass[3]

precision <- nnet_cm2$byClass[1]
recall <- nnet_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(nnet_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm2$byClass[2]
fp_rate <- gbm_cm2$byClass[3]

precision <- gbm_cm2$byClass[1]
recall <- gbm_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_2$Fraud, as.numeric(gbm_predictions2))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions2), as.numeric(reducedTest_2$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm2$byClass[2]
fp_rate <- weka_cm2$byClass[3]

precision <- weka_cm2$byClass[1]
recall <- weka_cm2$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Recursive feature elimination with SVM (Recursive Feature Elimination-SVM or RFE-SVM)
library(e1071)
svm_fit <- svm(Fraud ~ ., data = train, kernel = "linear")
rfe_svm_res <- rfe(x = train[, -which(names(train) %in% c("Fraud"))], y = train$Fraud, sizes = c(1:8), rfeControl = rfeControl(functions = rfFuncs), method = "svm", fit = svm_fit)
reducedTrain_3 <- train[, c("Fraud", rfe_svm_res$optVariables)]
reducedTest_3 <- test[, c("Fraud", rfe_svm_res$optVariables)]

#Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "glm", trControl = control)
svm_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "svmLinear", trControl = control)
rf_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "rf", trControl = control)
knn_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "kknn", trControl = control)
cart_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "rpart", trControl = control)
nnet_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "nnet", trControl = control)
gbm_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "gbm", trControl = control)
weka_model3 <- train(Fraud ~ ., data = reducedTrain_3, method = "J48", trControl = control)


# Make predictions on the test set and evaluate the models
glm_predictions3 <- predict(glm_model3, newdata = reducedTest_3)
glm_cm3 <- confusionMatrix(glm_predictions3, reducedTest_3$Fraud)
glm_cm3

svm_predictions3 <- predict(svm_model3, newdata = reducedTest_3)
svm_cm3 <- confusionMatrix(svm_predictions3, reducedTest_3$Fraud)
svm_cm3

rf_predictions3 <- predict(rf_model3, newdata = reducedTest_3)
rf_cm3 <- confusionMatrix(rf_predictions3, reducedTest_3$Fraud)
rf_cm3

knn_predictions3 <- predict(knn_model3, newdata = reducedTest_3)
knn_cm3 <- confusionMatrix(knn_predictions3, reducedTest_3$Fraud)
knn_cm3

cart_predictions3 <- predict(cart_model3, newdata = reducedTest_3)
cart_cm3 <- confusionMatrix(cart_predictions3, reducedTest_3$Fraud)
cart_cm3

nnet_predictions3 <- predict(nnet_model3, newdata = reducedTest_3)
nnet_cm3 <- confusionMatrix(nnet_predictions3, reducedTest_3$Fraud)
nnet_cm3

gbm_predictions3 <- predict(gbm_model3, newdata = reducedTest_3)
gbm_cm3 <- confusionMatrix(gbm_predictions3, reducedTest_3$Fraud)
gbm_cm3

weka_predictions3 <- predict(weka_model3, newdata = reducedTest_3)
weka_cm3 <- confusionMatrix(weka_predictions3, reducedTest_3$Fraud)
weka_cm3

##############################################################################################

tp_rate <- glm_cm3$byClass[2]
fp_rate <- glm_cm3$byClass[3]

precision <- glm_cm3$byClass[1]
recall <- glm_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(glm_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm3$byClass[2]
fp_rate <- svm_cm3$byClass[3]

precision <- svm_cm3$byClass[1]
recall <- svm_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(svm_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm3$byClass[2]
fp_rate <- rf_cm3$byClass[3]

precision <- rf_cm3$byClass[1]
recall <- rf_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(rf_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm3$byClass[2]
fp_rate <- knn_cm3$byClass[3]

precision <- knn_cm3$byClass[1]
recall <- knn_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(knn_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm3$byClass[2]
fp_rate <- cart_cm3$byClass[3]

precision <- cart_cm3$byClass[1]
recall <- cart_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(cart_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm3$byClass[2]
fp_rate <- nnet_cm3$byClass[3]

precision <- nnet_cm3$byClass[1]
recall <- nnet_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(nnet_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm3$byClass[2]
fp_rate <- gbm_cm3$byClass[3]

precision <- gbm_cm3$byClass[1]
recall <- gbm_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_3$Fraud, as.numeric(gbm_predictions3))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions3), as.numeric(reducedTest_3$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm3$byClass[2]
fp_rate <- weka_cm3$byClass[3]

precision <- weka_cm3$byClass[1]
recall <- weka_cm3$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Principal component analysis (PCA):
library(caret)
preProc <- preProcess(train[, -which(names(train) %in% c("Fraud"))], method = "pca", pcaComp = 8)
reducedTrain_4 <- predict(preProc, train)
reducedTrain_4$Fraud <- train$Fraud
reducedTest_4 <- predict(preProc, test)
reducedTest_4$Fraud <- test$Fraud

#Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "glm", trControl = control)
svm_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "svmLinear", trControl = control)
rf_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "rf", trControl = control)
knn_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "kknn", trControl = control)
cart_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "rpart", trControl = control)
nnet_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "nnet", trControl = control)
gbm_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "gbm", trControl = control)
weka_model4 <- train(Fraud ~ ., data = reducedTrain_4, method = "J48", trControl = control)


# Make predictions on the test set and evaluate the models
glm_predictions4 <- predict(glm_model4, newdata = reducedTest_4)
glm_cm4 <- confusionMatrix(glm_predictions4, reducedTest_4$Fraud)
glm_cm4

svm_predictions4 <- predict(svm_model4, newdata = reducedTest_4)
svm_cm4 <- confusionMatrix(svm_predictions4, reducedTest_4$Fraud)
svm_cm4

rf_predictions4 <- predict(rf_model4, newdata = reducedTest_4)
rf_cm4 <- confusionMatrix(rf_predictions4, reducedTest_4$Fraud)
rf_cm4

knn_predictions4 <- predict(knn_model4, newdata = reducedTest_4)
knn_cm4 <- confusionMatrix(knn_predictions4, reducedTest_4$Fraud)
knn_cm4

cart_predictions4 <- predict(cart_model4, newdata = reducedTest_4)
cart_cm4 <- confusionMatrix(cart_predictions4, reducedTest_4$Fraud)
cart_cm4

nnet_predictions4 <- predict(nnet_model4, newdata = reducedTest_4)
nnet_cm4 <- confusionMatrix(nnet_predictions4, reducedTest_4$Fraud)
nnet_cm4

gbm_predictions4 <- predict(gbm_model4, newdata = reducedTest_4)
gbm_cm4 <- confusionMatrix(gbm_predictions4, reducedTest_4$Fraud)
gbm_cm4

weka_predictions4 <- predict(weka_model4, newdata = reducedTest_4)
weka_cm4 <- confusionMatrix(weka_predictions4, reducedTest_4$Fraud)
weka_cm4

##############################################################################################

tp_rate <- svm_cm4$byClass[2]
fp_rate <- svm_cm4$byClass[3]

precision <- svm_cm4$byClass[1]
recall <- svm_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(svm_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm4$byClass[2]
fp_rate <- rf_cm4$byClass[3]

precision <- rf_cm4$byClass[1]
recall <- rf_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(rf_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm4$byClass[2]
fp_rate <- knn_cm4$byClass[3]

precision <- knn_cm4$byClass[1]
recall <- knn_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(knn_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm4$byClass[2]
fp_rate <- cart_cm4$byClass[3]

precision <- cart_cm4$byClass[1]
recall <- cart_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(cart_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm4$byClass[2]
fp_rate <- nnet_cm4$byClass[3]

precision <- nnet_cm4$byClass[1]
recall <- nnet_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(nnet_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm4$byClass[2]
fp_rate <- gbm_cm4$byClass[3]

precision <- gbm_cm4$byClass[1]
recall <- gbm_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_4$Fraud, as.numeric(gbm_predictions4))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions4), as.numeric(reducedTest_4$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm4$byClass[2]
fp_rate <- weka_cm4$byClass[3]

precision <- weka_cm4$byClass[1]
recall <- weka_cm4$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Random forest:
library(randomForest)
set.seed(123)
rfFit <- randomForest(Fraud ~ ., data = train, ntree = 500, importance = TRUE)
importance <- importance(rfFit)
top_vars <- row.names(importance)[order(importance[, "MeanDecreaseGini"], decreasing = TRUE)[1:8]]
reducedTrain_5 <- train[, c("Fraud", top_vars)]
reducedTest_5 <- test[, c("Fraud", top_vars)]

#Model Trainings
control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "glm", trControl = control)
svm_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "svmLinear", trControl = control)
rf_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "rf", trControl = control)
knn_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "kknn", trControl = control)
cart_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "rpart", trControl = control)
nnet_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "nnet", trControl = control)
gbm_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "gbm", trControl = control)
weka_model5 <- train(Fraud ~ ., data = reducedTrain_5, method = "J48", trControl = control)


# Make predictions on the test set and evaluate the models
glm_predictions5 <- predict(glm_model5, newdata = reducedTest_5)
glm_cm5 <- confusionMatrix(glm_predictions5, reducedTest_5$Fraud)
glm_cm5

svm_predictions5 <- predict(svm_model5, newdata = reducedTest_5)
svm_cm5 <- confusionMatrix(svm_predictions5, reducedTest_5$Fraud)
svm_cm5

rf_predictions5 <- predict(rf_model5, newdata = reducedTest_5)
rf_cm5 <- confusionMatrix(rf_predictions5, reducedTest_5$Fraud)
rf_cm5

knn_predictions5 <- predict(knn_model5, newdata = reducedTest_5)
knn_cm5 <- confusionMatrix(knn_predictions5, reducedTest_5$Fraud)
knn_cm5

cart_predictions5 <- predict(cart_model5, newdata = reducedTest_5)
cart_cm5 <- confusionMatrix(cart_predictions5, reducedTest_5$Fraud)
cart_cm5

nnet_predictions5 <- predict(nnet_model5, newdata = reducedTest_5)
nnet_cm5 <- confusionMatrix(nnet_predictions5, reducedTest_5$Fraud)
nnet_cm5

gbm_predictions5 <- predict(gbm_model5, newdata = reducedTest_5)
gbm_cm5 <- confusionMatrix(gbm_predictions5, reducedTest_5$Fraud)
gbm_cm5

weka_predictions5 <- predict(weka_model5, newdata = reducedTest_5)
weka_cm5 <- confusionMatrix(weka_predictions5, reducedTest_5$Fraud)
weka_cm5

##############################################################################################

tp_rate <- glm_cm5$byClass[2]
fp_rate <- glm_cm5$byClass[3]

precision <- glm_cm5$byClass[1]
recall <- glm_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(glm_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm5$byClass[2]
fp_rate <- svm_cm5$byClass[3]

precision <- svm_cm5$byClass[1]
recall <- svm_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(svm_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm5$byClass[2]
fp_rate <- rf_cm5$byClass[3]

precision <- rf_cm5$byClass[1]
recall <- rf_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(rf_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm5$byClass[2]
fp_rate <- knn_cm5$byClass[3]

precision <- knn_cm5$byClass[1]
recall <- knn_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(knn_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm5$byClass[2]
fp_rate <- cart_cm5$byClass[3]

precision <- cart_cm5$byClass[1]
recall <- cart_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(cart_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm5$byClass[2]
fp_rate <- nnet_cm5$byClass[3]

precision <- nnet_cm5$byClass[1]
recall <- nnet_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(nnet_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm5$byClass[2]
fp_rate <- gbm_cm5$byClass[3]

precision <- gbm_cm5$byClass[1]
recall <- gbm_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_5$Fraud, as.numeric(gbm_predictions5))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions5), as.numeric(reducedTest_5$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm5$byClass[2]
fp_rate <- weka_cm5$byClass[3]

precision <- weka_cm5$byClass[1]
recall <- weka_cm5$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

#Filter-based method using chi-squared test:
chiSquared <- apply(data[, -which(names(data) %in% c("Fraud"))], 2, function(x) chisq.test(table(x, data$Fraud))$statistic)
selected <- order(chiSquared, decreasing = TRUE)[1:8]
# Create reduced datasets using selected attributes
reducedTrain_6 <- train[, c("Fraud", names(train)[selected])]
reducedTest_6 <- test[, c("Fraud", names(test)[selected])]


control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "glm", trControl = control)
svm_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "svmLinear", trControl = control)
rf_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "rf", trControl = control)
knn_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "kknn", trControl = control)
cart_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "rpart", trControl = control)
nnet_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "nnet", trControl = control)
gbm_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "gbm", trControl = control)
weka_model6 <- train(Fraud ~ ., data = reducedTrain_6, method = "J48", trControl = control)

# Make predictions on the test set and evaluate the models
glm_predictions6 <- predict(glm_model6, newdata = reducedTest_6)
glm_cm6 <- confusionMatrix(glm_predictions6, reducedTest_6$Fraud)
glm_cm6

svm_predictions6 <- predict(svm_model6, newdata = reducedTest_6)
svm_cm6 <- confusionMatrix(svm_predictions6, reducedTest_6$Fraud)
svm_cm6

rf_predictions6 <- predict(rf_model6, newdata = reducedTest_6)
rf_cm6 <- confusionMatrix(rf_predictions6, reducedTest_6$Fraud)
rf_cm6

knn_predictions6 <- predict(knn_model6, newdata = reducedTest_6)
knn_cm6 <- confusionMatrix(knn_predictions6, reducedTest_6$Fraud)
knn_cm6

cart_predictions6 <- predict(cart_model6, newdata = reducedTest_6)
cart_cm6 <- confusionMatrix(cart_predictions6, reducedTest_6$Fraud)
cart_cm6

nnet_predictions6 <- predict(nnet_model6, newdata = reducedTest_6)
nnet_cm6 <- confusionMatrix(nnet_predictions6, reducedTest_6$Fraud)
nnet_cm6

gbm_predictions6 <- predict(gbm_model6, newdata = reducedTest_6)
gbm_cm6 <- confusionMatrix(gbm_predictions6, reducedTest_6$Fraud)
gbm_cm6

weka_predictions6 <- predict(weka_model6, newdata = reducedTest_6)
weka_cm6 <- confusionMatrix(weka_predictions6, reducedTest_6$Fraud)
weka_cm6

##############################################################################################

tp_rate <- glm_cm6$byClass[2]
fp_rate <- glm_cm6$byClass[3]

precision <- glm_cm6$byClass[1]
recall <- glm_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(glm_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm6$byClass[2]
fp_rate <- svm_cm6$byClass[3]

precision <- svm_cm6$byClass[1]
recall <- svm_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(svm_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm6$byClass[2]
fp_rate <- rf_cm6$byClass[3]

precision <- rf_cm6$byClass[1]
recall <- rf_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(rf_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm6$byClass[2]
fp_rate <- knn_cm6$byClass[3]

precision <- knn_cm6$byClass[1]
recall <- knn_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(knn_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm6$byClass[2]
fp_rate <- cart_cm6$byClass[3]

precision <- cart_cm6$byClass[1]
recall <- cart_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(cart_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm6$byClass[2]
fp_rate <- nnet_cm6$byClass[3]

precision <- nnet_cm6$byClass[1]
recall <- nnet_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(nnet_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm6$byClass[2]
fp_rate <- gbm_cm6$byClass[3]

precision <- gbm_cm6$byClass[1]
recall <- gbm_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_6$Fraud, as.numeric(gbm_predictions6))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions6), as.numeric(reducedTest_6$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm6$byClass[2]
fp_rate <- weka_cm6$byClass[3]

precision <- weka_cm6$byClass[1]
recall <- weka_cm6$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################
####################################################################################################

data$AC_1001_Issue <- as.numeric(data$AC_1001_Issue)
data$AC_1002_Issue <- as.numeric(data$AC_1002_Issue)
data$AC_1003_Issue <- as.numeric(data$AC_1003_Issue)
data$TV_2001_Issue <- as.numeric(data$TV_2001_Issue)
data$TV_2002_Issue <- as.numeric(data$TV_2002_Issue)
data$TV_2003_Issue <- as.numeric(data$TV_2003_Issue)
data$Claim_Value <- as.numeric(data$Claim_Value)
data$Service_Centre <- as.numeric(data$Service_Centre)
data$Product_Age <- as.numeric(data$Product_Age)
data$Fraud <- as.numeric(data$Fraud)
DF1 <- data[, c("AC_1001_Issue","AC_1002_Issue","AC_1003_Issue","TV_2001_Issue","TV_2002_Issue","TV_2003_Issue","Claim_Value","Service_Centre","Product_Age","Fraud")]

# Compute the correlation matrix
correlation_matrix <- cor(DF1)
# Extract the correlations with is_recid
correlations <- correlation_matrix[, "Fraud"]
# Sort the correlations in descending order
sorted_correlations <- sort(correlations, decreasing = TRUE)
# Print the sorted correlations
sorted_correlations <- sorted_correlations[-1]
print(sorted_correlations[1:4])

reducedTrain_7 <- train[c('Claim_Value', 'TV_2002_Issue', 'TV_2003_Issue', 'TV_2001_Issue', 'Fraud')]
reducedTrain_7$Fraud <- as.factor(reducedTrain_7$Fraud)
reducedTest_7 <- test[c('Claim_Value', 'TV_2002_Issue', 'TV_2003_Issue', 'TV_2001_Issue', 'Fraud')]
reducedTest_7$Fraud <- as.factor(reducedTest_7$Fraud)



control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, summaryFunction = defaultSummary)
glm_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "glm", trControl = control)
svm_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "svmLinear", trControl = control)
rf_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "rf", trControl = control)
knn_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "kknn", trControl = control)
cart_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "rpart", trControl = control)
nnet_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "nnet", trControl = control)
gbm_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "gbm", trControl = control)
weka_model7 <- train(Fraud ~ ., data = reducedTrain_7, method = "J48", trControl = control)


# Make predictions on the test set and evaluate the models
glm_predictions7 <- predict(glm_model7, newdata = reducedTest_7)
glm_cm7 <- confusionMatrix(glm_predictions7, reducedTest_7$Fraud)
glm_cm7

svm_predictions7 <- predict(svm_model7, newdata = reducedTest_7)
svm_cm7 <- confusionMatrix(svm_predictions7, reducedTest_7$Fraud)
svm_cm7

rf_predictions7 <- predict(rf_model7, newdata = reducedTest_7)
rf_cm7 <- confusionMatrix(rf_predictions7, reducedTest_7$Fraud)
rf_cm7

knn_predictions7 <- predict(knn_model7, newdata = reducedTest_7)
knn_cm7 <- confusionMatrix(knn_predictions7, reducedTest_7$Fraud)
knn_cm7

cart_predictions7 <- predict(cart_model7, newdata = reducedTest_7)
cart_cm7 <- confusionMatrix(cart_predictions7, reducedTest_7$Fraud)
cart_cm7

nnet_predictions7 <- predict(nnet_model7, newdata = reducedTest_7)
nnet_cm7 <- confusionMatrix(nnet_predictions7, reducedTest_7$Fraud)
nnet_cm7

gbm_predictions7 <- predict(gbm_model7, newdata = reducedTest_7)
gbm_cm7 <- confusionMatrix(gbm_predictions7, reducedTest_7$Fraud)
gbm_cm7

weka_predictions7 <- predict(weka_model7, newdata = reducedTest_7)
weka_cm7 <- confusionMatrix(weka_predictions7, reducedTest_7$Fraud)
weka_cm7

##############################################################################################

tp_rate <- glm_cm7$byClass[2]
fp_rate <- glm_cm7$byClass[3]

precision <- glm_cm7$byClass[1]
recall <- glm_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(glm_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(glm_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- svm_cm7$byClass[2]
fp_rate <- svm_cm7$byClass[3]

precision <- svm_cm7$byClass[1]
recall <- svm_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(svm_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(svm_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- rf_cm7$byClass[2]
fp_rate <- rf_cm7$byClass[3]

precision <- rf_cm7$byClass[1]
recall <- rf_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(rf_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(rf_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- knn_cm7$byClass[2]
fp_rate <- knn_cm7$byClass[3]

precision <- knn_cm7$byClass[1]
recall <- knn_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(knn_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(knn_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- cart_cm7$byClass[2]
fp_rate <- cart_cm7$byClass[3]

precision <- cart_cm7$byClass[1]
recall <- cart_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(cart_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(cart_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- nnet_cm7$byClass[2]
fp_rate <- nnet_cm7$byClass[3]

precision <- nnet_cm7$byClass[1]
recall <- nnet_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(nnet_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- gbm_cm7$byClass[2]
fp_rate <- gbm_cm7$byClass[3]

precision <- gbm_cm7$byClass[1]
recall <- gbm_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(reducedTest_7$Fraud, as.numeric(gbm_predictions7))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(gbm_predictions7), as.numeric(reducedTest_7$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

##############################################################################################

tp_rate <- weka_cm7$byClass[2]
fp_rate <- weka_cm7$byClass[3]

precision <- weka_cm7$byClass[1]
recall <- weka_cm7$byClass[2]

f_measure <- (2 * precision * recall) / (precision + recall)

# Calculate ROC area
library(pROC)
roc <- roc(test$Fraud, as.numeric(nnet_predictions))
roc_area <- auc(roc)

# Calculate MCC
mcc <- cor(as.numeric(nnet_predictions), as.numeric(test$Fraud), method = "pearson")

print(paste('True Positive Rate: ',tp_rate))
print(paste('False positive Rate: ', fp_rate))
print(paste('Precision: ', precision))
print(paste('Recall: ', recall))
print(paste('F-measure: ', f_measure))
print(paste('ROC value: ', roc_area))
print(paste('MCC value: ', mcc))

# Create a named vector with the performance metrics
perf_Weka <- c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision, recall = recall, 
               f_measure = f_measure, roc_area = roc_area, mcc = mcc)

# Print the named vector
print(perf_Weka)

####################################################################################################

#Export Datasets

write.csv(train, file = "train.csv")
write.csv(test, file = "test.csv")
write.csv(reducedTrain_3, file = "Reduced_train.csv")
write.csv(reducedTest_3, file = "Reduced_test.csv")

