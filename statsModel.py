# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:32:26 2019

@author: paul.avendano
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn
## loads Boston house prices dataset from datasets library 
data = datasets.load_boston() 

#prints the desciption of the dataset
print (data.DESCR)
"""
      - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM*       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT*    % lower status of the population
        - MEDV*     Median value of owner-occupied homes in $1000's
"""
"""
print the column names of the independent 
variables and the dependent variable, respectively

Scikit-learn has already set the house value/price data as a target 
variable and 13 other variables are set as predictors
"""
data.feature_names
data.target

# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
#MEDV is $1000s
target = pd.DataFrame(data.target, columns=["MEDV"])

#Taking RM and LSTAT to fit the regression model, using no constant (is not added by default)
X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
"""
OLS: ordinary least squares, fit a regression line that would minimize the 
square of distance from the regression line
"""
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

#prints the sumary data of the prediction model 
model.summary()

#adding a constant
X = df["RM"] ## X usually means our input variables (or independent variables)
y = target["MEDV"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

