# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:23:05 2018
Simple Linear Regression Model to find correlation between Employee Salary & Years of Experience 
and predict best salaries for future employees
@author: Priyanka Sharma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#removing the salary column to only pick the independent variable - years of experience
X = dataset.iloc[:, :-1].values 
#pick the dependent variable - Salary
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling - not needed for Simple Linear Regression as the library will take care of the same

#Fitting Simple Linear Regression Model to Training Set
from sklearn.linear_model import LinearRegression
#create object of the class
regressor = LinearRegression()
#fit the object to the training set
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
#comparing real salaries to the predicted salaries
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")￼
plt.show()

#Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
#comparing real salaries to the predicted salaries
#no need to change below since regressor is already trained on training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#EOF















