# import packages
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# import logisitic regression model
from sklearn.linear_model import LogisticRegression

# create the LR model object
cls = LogisticRegression(max_iter=2000)

# To avoid this warning/error:
# ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#
# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(

# You can increase the max_iter as suggested, like so:
# cls = LogisticRegression(max_iter=2000)

# import data or load data
data = load_breast_cancer()

# view data to modify or use as it is by target (view of first 5 rows)
view = data.data[0:5]

# do needed update on data if needed (in this case, no changes are necessary)

# inform program about training and test data (data is split in the ratio 8:2)
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    train_size=0.8, random_state=50
)

# fit data
cls = cls.fit(X_train, y_train)

# let the model make predictions
predict = cls.predict(X_test)

# create and verify accuracy with confusion matrix
cm = confusion_matrix(y_test, predict)

print("Accuracy: ", accuracy_score(y_test, predict))
# confusion matrix (text representation)
print(cm)

# plot figures

# confusion matrix (visual representation)
ConfusionMatrixDisplay.from_predictions(
        y_test,
        predict,
        display_labels=["Negative", "Positive"]
)
plt.title("Breast Cancer Prediction using Logistic Regression")
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.show()
