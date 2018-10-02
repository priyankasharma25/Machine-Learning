# -*- coding: utf-8 -*-
"""
Created on Sat Oct 02 17:12:25 2018

Polynomial Regression Model to help HR department of an organization find 
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
# Independant Variable
X = dataset.iloc[:, 1:2].values
# Dependant Variable
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# Not required since we do not have much data so we cannot split
# Also, we need to use all data to make accurate prediction
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# Not required as it is taken care of by the library
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
# added for comparison with Polynomial Regression Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# below code transforms X into new matrix containing independent variables with different powers
# degree can be adjusted to see which model fits better
poly_reg = PolynomialFeatures(degree =4) 
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing Linear Regression results
# added for comparison with Polynomial Regression Model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing Polynomial Regression results
# X_grid contains all the levels along with incremented levels with resolution of 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
# added for comparison with Polynomial Regression Model
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))


























