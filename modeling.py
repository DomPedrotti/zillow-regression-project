import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoCV
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

import split_scale as ss
import evaluate as ev
import features as fe

def plot_residuals(x,y, dataframe = None):
    '''
    plot_residuals(x,y, dataframe = None)

    renders seaborn residual plot
    
    args:
    x: string independant variable name or pandas Series
    y: string target variable name or pandas Series
    dataframe: optional pandas dataframe, used if x and y are column names


    returns None
    '''
    sns.residplot(x,y,data = dataframe)


def plot_regression(x, y):
    sns.relplot(x,y)
    pass

def model_predictions(x_train, y_train, x_test, y_test, model_type = LinearRegression()):
    #fit model_object
    model_object = model_type.fit(x_train, y_train)
    
    yhat = model_object.predict(x_test)
    
    results = x_test.join(y_test)
    results['predicted'] = yhat
    return results