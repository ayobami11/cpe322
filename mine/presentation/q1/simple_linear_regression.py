from pandas import read_csv

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read data from a sample diabetes dataset
salary_data = read_csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/salary.csv")

# Split data into features (X) and labels (y)
X = salary_data[salary_data.columns[1:2]].values
y = salary_data[salary_data.columns[-1]].values

# Split training and test data using the ratio 8:2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=25)

# Instantiate the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions using the model
y_pred = lr_model.predict(X_test)

# - Evaluate the performance of the model
print("Simple Linear Regression Results\n")
print(f"Intercept: \n {lr_model.intercept_}")
print(f"Coefficient: \n {lr_model.coef_}")
print(f"Mean absolute error: \n {mean_squared_error(y_test, y_pred)}")
print(f"Mean squared error: \n {mean_absolute_error(y_test, y_pred)}")