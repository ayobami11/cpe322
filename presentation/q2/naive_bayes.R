# install the following packages
# c function creates a character vector containing the desired strings as items
install.packages(c("caret", "e1071"))
library(caret)
library(e1071)

# Read data into the 'diabetes_data' variable
diabetes_data <- read.csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/diabetes.csv", sep = ",", header = TRUE)
print(diabetes_data)

# split data into training and testing datasets

# Generates a randomized sample of size (number of data samples) containing 0s and 1s using the ratio 8:2
set.seed(3)
id <- sample(0:1, nrow(diabetes_data), prob = c(0.8, 0.2), replace = TRUE)
# Data samples which map to the 0 value from the sample are assigned to the training dataset
training_dataset = diabetes_data[id == 0,]
View(training_dataset)
# Data samples which map to the 1 value from the sample are assigned to the testing dataset
testing_dataset = diabetes_data[id == 1,]
View(testing_dataset)


# checking the dimensions of training and testing dataframes
dim(training_dataset);
dim(testing_dataset);


# checks for any null (or non-available) values
anyNA(diabetes_data)

summary(diabetes_data)


diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)
training_dataset$Outcome <- as.factor(training_dataset$Outcome)


# train the model
nb_model <- naiveBayes(Outcome ~ ., data = training_dataset)

# A-prior probabalites refer to the prior probabilities
# Fields under conditional probabilities contain the likelihood tables for each feature

# [, 1] represents the average (mean) values for 0 = no_diabetes and 1 = diabetes
# [, 2] represents the standard deviation (variability) values for 0 = no_diabetes and 1 = diabetes
nb_model

# Make predictions using the model
test_pred <- predict(nb_model, testing_dataset)
test_pred

# confusion matrix
# 95% CI = 95% confident that accuracy lies within the range (0.5793, 0.7347)
# accuracy = (TP + TN) / (TP + FP + TN + FN)

# false positives = number of incorrect predictions when the actual value is positive
# false negatives = number of incorrect predictions when the actual value is negative

#                               Actual values
#    Predicted values             0                  1
#         0              true_negatives    false_negatives
#         1              false_positives   true_positives

# Evaluate the performance of the model
confusionMatrix(table(test_pred, testing_dataset$Outcome))