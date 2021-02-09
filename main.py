"""
main.py
Author: Brando
Created: 2/1/2021 7:36:44 PM

Task 1: Fitting linear regression to quadratic data: Apply linear regression to the following dataset (Links to an external site.) (where we are trying to use column X to predict Y) and plot the datapoints with the line of best fit. See if you can apply some feature engineering to fit the data perfectly.

Task 2: Apply logistic regression on the MNIST train dataset and tabulate the precision and recall of the different digits in the test dataset.

References:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=LinearRegression%20fits%20a%20linear%20model,the%20intercept%20for%20this%20model.

fit() trains the model 
"""
### Modules ###
from operator import mod
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris, fetch_openml, load_boston, load_digits
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os.path 
from os import path 
import pickle
import bunch
from sklearn.metrics import confusion_matrix, classification_report

### Task 1 ### 
# iris = load_iris()
# linearRegressionHandler = LinearRegression()
# sampleData = pd.read_csv("B:\\COLLEGE\\20_21\\Spring21\\CES514\\Homework\\Week2\\sample_data.csv", index_col=0) # use index_col to tell which column has the row index 


# x_train, x_test, y_train, y_test = train_test_split(sampleData['X'], sampleData['Y'], test_size=0.2, random_state=0)

# x_train = x_train.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)
# x_test = x_test.reshape(-1, 1)

# linearRegressionHandler.fit(x_train, y_train)

# result = linearRegressionHandler.predict(x_test)

# plt.plot(result,'.')
# plt.show()

### Task 2 ### 
mnistFile = "mnist.cache"
logisticRegressionHandler = LogisticRegression()
dataArray = np.zeros(1) # initialize the variable 
numberWeWant = ['5','6']
digitIndexArray = np.zeros(1) 
maxNumbersToFit = 200

# Get the mnist data set 
# if the mnist cache exists, then read that file
if path.exists(mnistFile):
    mnist = pickle.load(open(mnistFile, "rb"))[0]

# otherwise fetch the dataset and create the cache 
else: 
    mnist = fetch_openml('mnist_784', version=1)
    pickle.dump([mnist], open(mnistFile, "wb"))

# attempt to lower the iterations and complexity 
# digitDataFrame = pd.DataFrame({"Digits":mnist.target[:]})
targetData = pd.DataFrame({"Digits":mnist.target[:maxNumbersToFit]})
digitIndexArray = targetData.index
sourceData = mnist.data.loc[digitIndexArray]
logisticRegressionHandler.fit(sourceData, pd.to_numeric(targetData['Digits']))

# do next two hundred for testing
testTargetData = pd.DataFrame({"Digits":mnist.target[maxNumbersToFit:(2*maxNumbersToFit)]})
testSourceData = mnist.data.loc[testTargetData.index]
prediction = logisticRegressionHandler.predict(testSourceData)
print(prediction)
print(confusion_matrix(pd.to_numeric(testTargetData['Digits']), prediction))
print(classification_report(pd.to_numeric(testTargetData['Digits']), prediction))

# Plot the number image 
# dataArray = np.array(mnist.data) # Put in numpy array for image display 
# plt.imshow(dataArray[0].reshape((28,28)))
# plt.show()
