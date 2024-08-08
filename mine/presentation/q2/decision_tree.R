# Download and install the required packages

# rpart (recursive partitioning) library is used for building decision tree
install.packages(c("caret", "rpart"))
library(caret)
library(rpart)

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


diabetes_data$Outcome <- as.factor(diabetes_data$Outcome)
training_dataset$Outcome <- as.factor(training_dataset$Outcome)

dt_model = rpart(Outcome~., data = training_dataset)
dt_model

# Make predictions using the model
test_pred <- predict(dt_model, newdata = testing_dataset, type = "class")
test_pred

# Evaluate the performance of the model
confusionMatrix(table(test_pred, testing_dataset$Outcome))