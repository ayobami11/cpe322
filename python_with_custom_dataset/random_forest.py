# import packages
from pandas import read_csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# import random forest model
from sklearn.ensemble import RandomForestClassifier

# create the random forest model
cls = RandomForestClassifier()

# import data or load data
data = read_csv("./python_with_custom_dataset/diabetes.csv")

# Split data into features (X) and labels (y)
X = data[data.columns[:-1]].values
y = data[data.columns[-1]].values

# view data to modify or use as it is by target (view of first 5 rows)
view = data.head()

# to update as required (in this case, no changes are necessary)

# inform program about training and test data (data is split in the ratio 8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)

# fit the model
cls = cls.fit(X_train, y_train)

# predict outcome
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
plt.title("Breast Cancer Prediction using Random Forest")
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.show()
