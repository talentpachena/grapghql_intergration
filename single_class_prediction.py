# example of training a final classification mode
# If you had just one new data instance,
# you can provide this as instance wrapped in an
# array to the predict() function
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# define one new instance
Xnew = [[-0.79415228, 2.10495117]]
# make a prediction
ynew = model.predict(Xnew)
print("X={}, Predicted={}".format(Xnew[0], ynew[0]))