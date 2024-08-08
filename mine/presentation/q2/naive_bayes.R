# install the following packages
install.packages("e1071")
library(e1071)

# Read data into the 'diabetes_data' variable
diabetes_data <- read.csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/diabetes.csv", sep = ",", header = TRUE)
View(diabetes_data)

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

# train the model
nb_model <- naiveBayes(Outcome ~ ., data = training_dataset)
nb_model

# Make predictions using the model
test_pred <- predict(nb_model, testing_dataset)
test_pred

# confusion matrix
# 95% CI = 95% confident that accuracy lies within the range (0.5793, 0.7347)
# accuracy = (TP + TN) / (TP + FP + TN + FN)

# Evaluate the performance of the model
confusionMatrix(table(test_pred, testing_dataset$Outcome))