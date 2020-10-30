# make is the probability of the data instance belonging to each class.
# given a new instance,
# the model returns the probability for 
# each outcome class as a value between 0 and 1
# You can make these types of predictions in 
# scikit-learn by calling the predict_proba() function
# below model below makes a probability prediction for
# each example in the Xnew array of data instance.

# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
for i in range(len(Xnew)):
    print("X={}, Predicted={}".format(Xnew[i], ynew[i]))