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
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, fetch_openml, load_boston
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import os.path 
from os import path 
import pickle
import bunch

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
mnistFile = "mnist.p"
# mnist = fetch_openml('mnist_784', version=1)

# pickle.dump([mnist], open(mnistFile, "wb"))

mnistLoad = pickle.load(open(mnistFile, "rb"))[0]
# mnist = Bunch(mnistLoad)
print(mnistLoad.target)