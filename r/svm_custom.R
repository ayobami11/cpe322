# install packages
install.packages(c("caret", "e1071"))

# load the installed packages into the R environment
library(caret)

# import the model special package library (e1071)
library(e1071)

# load data by input at text base
data("iris")

# check and correct data
summary(iris)

# split data into training and test data, and assign to variables
set.seed(500)
id <- sample(0:1, nrow(iris), prob = c(0.8, 0.2), replace = TRUE)

train = iris[id == 0,]
test = iris[id == 1,]

# assign data to program
X_train <- train
Y_train <- train$Species
X_test <- test

# fit data to model
svm_model <- svm(formula = Species ~ ., data = iris, kernel = "linear", type = "C-classification")

# predict
predict = predict(svm_model, newdata = test)

# print outcome
print(predict)