# install packages
install.packages("caret")

# load the installed packages into the R environment
library(caret)

# import model
# model = lm()

# load data by input at text base
data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/mine/presentation/diabetes.csv", sep = ",", header = TRUE)

# check and correct data
summary(data)

# split data into training and test data, and assign to variables
set.seed(100)
id <- sample(0:1, nrow(data), prob = c(0.8, 0.2), replace = TRUE)

train = data[id == 0,]
test = data[id == 1,]

X_train <- train
Y_train <- test
X_test <- train$Outcome

# fit data to model
model <- lm(Outcome ~ ., data = data)

# predict
pred = predict(model, newdata = test)

# print outcome
print(pred)