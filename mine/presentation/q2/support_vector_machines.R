# Download and install the required packages
install.packages("caret")
library(caret)

# Read data into the 'diabetes_data' variable
diabetes_data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/mine/presentation/diabetes.csv", sep = ",", header = TRUE)
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

# converts the outcome (target) column to a factor variable
# a factor categorizes and stores data using a limited number of values
training_dataset$Outcome <- as.factor(training_dataset$Outcome)

# trains data on different algorithm using svmLinear
svm_linear <- train(Outcome ~., data = training_dataset, method = "svmLinear")
svm_linear

# Make predictions using the model
test_pred <- predict(svm_linear, newdata = testing_dataset)
test_pred

# Evaluate the performance of the model
confusionMatrix(table(test_pred, testing_dataset$Outcome))