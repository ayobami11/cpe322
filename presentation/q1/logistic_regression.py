from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Read data from a sample diabetes dataset
diabetes_data = read_csv("/Users/tunwa/Documents/coding-files/cpe322/presentation/diabetes.csv")

# Split data into features (X) and labels (y)
X = diabetes_data[diabetes_data.columns[:-1]].values
y = diabetes_data[diabetes_data.columns[-1]].values

#  Standardizes features by removing the mean and scaling to unit variance i.e. mean = 0 sd = 1
scaler = StandardScaler()
#  Computes the mean and standard deviation to be used for feature scaling
scaler.fit(X)

# Performs standardization by centering and scaling
X_scaled = scaler.transform(X)

# Split training and test data using the ratio 8:2
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, random_state=25)

# Instantiate the model 
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train_scaled, y_train)

# Make predictions using the model
y_pred = lr_model.predict(X_test_scaled)

# - Evaluate the performance of the model

print("Logisitic Regression Results\n")
print(f"Classification report: \n{classification_report(y_test, y_pred)}\n")
print(f"Accuracy: \n{accuracy_score(y_test, y_pred)}\n")
print(f"Confusion matrix: \n{confusion_matrix(y_test, y_pred)}\n")

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#                               Predicted values
#    Actual values             0                  1
#         0              true_negatives    false_positives
#         1              false_negatives   true_positives