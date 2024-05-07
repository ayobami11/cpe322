install.packages(c("caret", "randomForest"))
library(caret)
library(randomForest)

# Read data into the 'diabetes_data' variable
diabetes_data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/diabetes.csv", sep = ",", header = TRUE)
View(diabetes_data)

# Generates a randomized sample of size (number of data samples) containing 0s and 1s using the ratio 8:2
set.seed(3)
id <- sample(0:1, nrow(diabetes_data), prob = c(0.8, 0.2), replace = TRUE)
# Data samples which map to the 0 value from the sample are assigned to the training dataset
training_dataset = diabetes_data[id == 0,]
View(training_dataset)
# Data samples which map to the 1 value from the sample are assigned to the testing dataset
testing_dataset = diabetes_data[id == 1,]
View(testing_dataset)

diabetes_data$Outcome <- factor(diabetes_data$Outcome)
training_dataset$Outcome <- factor(training_dataset$Outcome)

bestmtry <- tuneRF(training_dataset, training_dataset$Outcome, stepFactor = 1.2, improve = 0.01, trace = TRUE, plot = TRUE)

rf_model <- randomForest(Outcome~., data = training_dataset)
rf_model

# gives Gini index (priority of variables)
importance(rf_model)

varImpPlot(rf_model)

# Make predictions using the model
test_pred <- predict(rf_model, newdata = testing_dataset, type = "class")
test_pred

confusionMatrix(table(test_pred, testing_dataset$Outcome))