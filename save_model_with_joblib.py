import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib # is part of the SciPy ecosystem and provides utilities for pipelining Python jobs.
              # It provides utilities for saving and loading Python objects that make use
              # of NumPy data structures, efficiently.
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
# This can be useful for some machine learning algorithms that require a lot of parameters
# or store the entire dataset (like K-Nearest Neighbors).
filename = 'finalized_model.sav'
joblib.dump(model, filename)
 
# some time later...
 
# load the model from disk
# an estimate of the accuracy of the model on unseen data is reported.
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)