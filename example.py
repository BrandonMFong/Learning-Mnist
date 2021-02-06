
### Modules ###
from operator import mod
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import pprint
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import train_test_split
boston = load_boston()

df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

model = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state = 4)
model.fit(x_train, y_train)

result = model.predict(x_test) # result should be what?
# print(y_test)
plt.plot(x_test,'.')
# plt.plot(x_test,'.')
plt.show()