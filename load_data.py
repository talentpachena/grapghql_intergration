from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print(dataset)

# get dimensions of the dataset
# shape
print(dataset.shape)

# Peek at the Data
# first 20 items
# head
print(dataset.head(20))

# get statistical summary of the dataset
# descriptions
print(dataset.describe())

# Class Distribution
# number of instances (rows) belong to each class
# class distribution
print(dataset.groupby('class').size())
print("")
print("")
print("Data Visualization")

# Univariate plots to better understand each attribute.
# plots of each individual variable.
# Given that the input variables are numeric,
# we can create box and whisker plots of each.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show() # This gives us a much clearer idea of the distribution of the input attributes:

# a histogram of each input variable to get an idea of the distribution.
# histograms
dataset.hist()
pyplot.show()

print("Multivariate Plots")
# Now we can look at the interactions between the variables.
# look at scatterplots of all pairs of attributes.
# This can be helpful to spot structured 
# relationships between input variables.
# scatter plot matrix
# the grouping of some pairs of attributes
# suggestscorrelation and a predictable relationship.
scatter_matrix(dataset)
pyplot.show()

print("Evaluate Some Algorithms")
# Create a Validation Dataset
# split the loaded dataset into two,
# 80% of which we will use to train,
# evaluate and select among our models,
# and 20% that we will hold back as a validation dataset.
# Split-out validation dataset
# data in the X_train and Y_train is for preparing models
# and a X_validation and Y_validation sets that we can use later. 
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# build models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
# compare the spread and the mean accuracy of each model. 
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset
# choose an algorithm to use to make predictions.
# fit the model on the entire training dataset and
# make predictions on the validation dataset.
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions)) #  confusion matrix provides an indication of the errors made.
print(classification_report(Y_validation, predictions)) # provides a breakdown of each class by precision