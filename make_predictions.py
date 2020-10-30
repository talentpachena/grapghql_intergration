# predict classification or regression outcomes
# with scikit-learn models in Python.
# Once you choose and fit a final machine learning model in scikit-learn,
# you can use it to make predictions on new data instances.
# Predict With Classification Models
# Predict With Regression Models

# Predict With Classification Models
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)

# we have one or more data instances in an array called Xnew
# This can be passed to the predict() function on our model in
# order to predict the class values for each instance in the array.
Xnew = [[4,2], [4,3]]
ynew = model.predict(Xnew)
print(Xnew)
print(ynew)
