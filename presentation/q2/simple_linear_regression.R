install.packages("caTools")
library(caTools)

# Read data into the 'salary_data' variable
salary_data = read.csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/salary.csv", sep=",", header = TRUE)
View(salary_data)

# Generates a randomized sample of size (number of data samples) containing 0s and 1s using the ratio 8:2
set.seed(3)
id <- sample(0:1, nrow(salary_data), prob = c(0.8, 0.2), replace = TRUE)
# Data samples which map to the 0 value from the sample are assigned to the training dataset
training_dataset = salary_data[id == 0,]
View(training_dataset)
# Data samples which map to the 1 value from the sample are assigned to the testing dataset
testing_dataset = salary_data[id == 1,]
View(testing_dataset)

lm.r = lm(formula = Salary ~ YearsExperience, data = training_dataset)

test_pred <- predict(lm.r, testing_dataset)

test_pred

summary(lm.r)