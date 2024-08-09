# install packages
install.packages("partykit")

# load the installed packages into the R environment
library(partykit)

# import model
# model = ctree()

# load data by input at text base
data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/mine/presentation/diabetes.csv", sep = ",", header = TRUE)

# check and correct data
summary(data)

# split data into training and test data, and assign to variable
set.seed(200)
id <- sample(0:1, nrow(data), prob = c(0.8, 0.2), replace = TRUE)

train = data[id == 0,]
test = data[id == 1,]

# assign data to program
X_train <- train
Y_train <- train$Outcome
X_test <- test

# fit data to model
model <- ctree(Outcome ~ ., data = data)
            
# predict
pred = predict(model, newdata = test)

# print outcome
print(pred)