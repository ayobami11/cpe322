# Download and install the required packages
install.packages("caret")
library(caret)

# Read data into the 'diabetes_data' variable
diabetes_data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/diabetes.csv", sep=",", header = TRUE)
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


# (generalized linear model) glm - used for creating a model in logistic regression
lr_model = glm(Outcome~., training_dataset, family = "binomial")
summary(lr_model)

# Significance code: *** - 99.9%; ** - 99%; * - 95%; . - 90%
# null deviance = deviation from the actual values of the dataset when using only the constant (beta 0)
# null deviance = deviation from the actual values of the dataset when using independent variables (beta n)

# Make predictions using the model
test_pred <- predict(lr_model, testing_dataset, type = "response")
test_pred


# Evaluate the performance of the model
confusionMatrix(table(as.numeric(test_pred > 0.5), testing_dataset$Outcome))