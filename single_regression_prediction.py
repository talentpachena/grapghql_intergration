# example of training a final classification mode
# The same function can be used to make a prediction
# for a single data instance as long as
# it is suitably wrapped in a surrounding list or array.
# the functions demonstrated for making regression predictions apply to
# all of the regression models available in scikit-learn.
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
# fit final model
model = LinearRegression()
model.fit(X, y)
# define one new data instance
Xnew = [[-1.07296862, -0.52817175]]
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
print("X={}, Predicted={}".format(Xnew[0], ynew[0]))