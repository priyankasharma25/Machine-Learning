# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:57:27 2018

Support Vector Regression Model to help HR department of an organization find 
if the previous salary quoted by the potential future employee is correct (truth or bluff) 
based on the designation-wise salary data of the past employer

@author: Priyanka Sharma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values # changed index from 2 to 2:3 to solve error in feature scaling 

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# needed for this model since SVR class doesn't apply feature scaling automatically
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Kernel = rbf is the gaussian kernal and is used for non linear regression
regressor.fit(X, y)

# Predicting a new result
# inverse_transform method is used since the data is scaled
# np arrary is used since it is the expected datatype of the transform method
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([(6.5,)])))) # arrary of 1x1

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
