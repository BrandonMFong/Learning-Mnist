"""
main.py
Author: Brando
Created: 2/1/2021 7:36:44 PM

References:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#:~:text=LinearRegression%20fits%20a%20linear%20model,the%20intercept%20for%20this%20model.

"""
### Modules ###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn import linear_model
from os import path 
import pickle
from sklearn.metrics import confusion_matrix, classification_report

### Task 1 ### 
def task1():
    """
    Task 1: Fitting linear regression to quadratic data: Apply linear regression to the following dataset (Links to an external site.) (where we are trying to use column X to predict Y) and plot the datapoints with the line of best fit. See if you can apply some feature engineering to fit the data perfectly.
    """
    linearRegressionHandler = LinearRegression()
    sampleData = pd.read_csv("B:\\COLLEGE\\20_21\\Spring21\\CES514\\Homework\\Week2\\sample_data.csv", index_col=0) # use index_col to tell which column has the row index 
    sampleData['X2'] = sampleData['X']**2
    linearRegressionHandler.fit(sampleData[['X2','X']], sampleData['Y'])
    prediction = linearRegressionHandler.predict(sampleData[['X2','X']])
    plt.plot(prediction,'.')
    plt.plot(sampleData['Y'],'.')
    plt.show()

### Task 2 ### 
def task2():
    """
    Task 2: Apply logistic regression on the MNIST train dataset and tabulate the precision and recall of the different digits in the test dataset.
    """
    mnistFile = "mnist.cache"
    logisticRegressionHandler = LogisticRegression()
    dataArray = np.zeros(1) # initialize the variable 
    numberWeWant = ['5','6']
    digitIndexArray = np.zeros(1) 
    maxNumbersToFit = 200

    print("*** PLEASE NOTE THAT I PUT A MAX ITERATION LIMIT TO LOWER THE COMPUTATIONS IN THIS SCRIPT ***\n")
    print("Max iterations = ", maxNumbersToFit, '\n')

    # Get the mnist data set 
    # if the mnist cache exists, then read that file
    if path.exists(mnistFile):
        mnist = pickle.load(open(mnistFile, "rb"))[0]

    # otherwise fetch the dataset and create the cache 
    else: 
        mnist = fetch_openml('mnist_784', version=1)
        pickle.dump([mnist], open(mnistFile, "wb"))

    # attempt to lower the iterations and complexity 
    targetData = pd.DataFrame({"Digits":mnist.target[:maxNumbersToFit]})
    digitIndexArray = targetData.index
    sourceData = mnist.data.loc[digitIndexArray]
    logisticRegressionHandler.fit(sourceData, pd.to_numeric(targetData['Digits']))

    # do next two hundred for testing
    testTargetData = pd.DataFrame({"Digits":mnist.target[maxNumbersToFit:(2*maxNumbersToFit)]})
    testSourceData = mnist.data.loc[testTargetData.index]
    prediction = logisticRegressionHandler.predict(testSourceData)
    print(confusion_matrix(pd.to_numeric(testTargetData['Digits']), prediction))
    print(classification_report(pd.to_numeric(testTargetData['Digits']), prediction))
    comparison = pd.DataFrame({"Data index" : testTargetData.index, "Real values" : mnist.target[maxNumbersToFit:(2*maxNumbersToFit)], "Prediction" : prediction})
    # comparison['Result'] = comparison['Real Values'] == comparison['Prediction']
    # print(comparison['Real Values'] == comparison['Prediction'])
    
    # dataArray = np.array(mnist.data) # Put in numpy array for image display 
    # plt.imshow(dataArray[396].reshape((28,28)))
    # plt.show()


if __name__ == '__main__':
    task1()
    task2()