# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:12:25 2018

Multiple Linear Regression Model to help venture capitalists find correlation between 
Profits & funds spent across different departments (eg R&D, Admin, Marketing etc) of 
50 startup companies operating in different states 
& predict Profits knowing the funds spent on different departments

@author: Priyanka Sharma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# Index of independent variable (or excluding dependent variable)
X = dataset.iloc[:, :-1].values
# Index of dependent variable
y = dataset.iloc[:, 4].values

# Encoding independant categorical varibale (State)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# index of the categorical variable column to be encoded
X[:,3] = labelencoder_X.fit_transform(X[:,3]) 
onehotencoder = OneHotEncoder(categorical_features =[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoing the dummy variable trap: removing 1 dummy variable manually
# this is to ensure that the dataset does not contain redundancies
# this is a sample code, the python library for Linear Regression takes care of it automatically
X = X[:,1:]
                   
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# no need to do manually as the library takes care of this step automatically

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
# Create object of the class
regressor = LinearRegression()
# Fit the object to the training set
regressor.fit(X_train, y_train)

# Predicting test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# below library is used to compute the statistical significance & evaluate the p-values
import statsmodels.formula.api as sm
# this library does not consider the constant b0, unlike other libraries
# hence we need to manually add a column of 1s to the dataset (consider b0 = b0 X0, where X0= 1)
# y = b0 + b1X1 +...+ bn Xn

# X = np.append(arr = X, values = np.ones((50,1)).astype(int),axis = 1) 
#axis = 1 if for adding column, axis = 0 is for adding rows

# above line adds new column at the end of the array, below line adds it to the beginning
X = np.append(arr = np.ones((50,1)).astype(int), values = X ,axis = 1)

# Start Backward Elimination
# creating a matrix containing only te independant variables that have high impact on the profit
# start by taking all columns (all indexes), we will then remove indexes recursively
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Step 2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Step 3 returns summary table
regressor_OLS.summary()
# Step 4 remove the predictor, independant variable with highest p-value
X_opt = X[:, [0, 1, 3, 4, 5]]
# Step 5 fith the model without the variable
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# repeat the steps 3 to 5 recursively
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()






















