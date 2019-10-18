from pydataset import data
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.metrics import explained_variance_score
import seaborn as sns
import math

def plot_residuals(x,y, dataframe = None):
    '''
    plot_residuals(x,y)

    renders seaborn residual plot
    
    args:
    x: string independant variable name or pandas Series
    y: string target variable name or pandas Series
    dataframe: optional pandas dataframe, used if x and y are column names

    returns None
    '''
    sns.residplot(x,y,data = dataframe)

def regression_errors(y,yhat):
    '''
    regression_errors(y,yhat)

    uses y and yhat series to calculate sum of squared errors, explained sum of squares, total sum of squares, mean squared error, and root mean squared error

    args:
    y: pandas series target variable
    yhat: pandas series calculated regression y variable

    returns:
    dictionary of floats 
    {'sse': sse, 'ess': ess, 'tss': tss, 'mse' : mse, 'rmse' : rmse }
    '''
    sse = ((yhat - y)**2).sum()
    ess = ((yhat - y.mean())**2).sum()
    tss = sse + ess
    mse = sse/len(y)
    rmse = math.sqrt(mse)
    return {'sse': sse, 'ess': ess, 'tss': tss, 'mse' : mse, 'rmse' : rmse }

def baseline_mean_errors(y):
    '''
    baseline_mean_errors(y)

    calculates sum of squared error, mean squared error, and root mean squared errors assuming no correlation

    args: 
    y: pandas series independant variable

    returns: dictionary of float values
    {'sse': sse, 'mse' : mse, 'rmse' : rmse }
    '''
    yhat = y.mean()
    sse = ((yhat - y)**2).sum()
    mse = sse/len(y)
    rmse = math.sqrt(mse)
    return {'sse': sse, 'mse' : mse, 'rmse' : rmse }

def better_than_baseline(y, yhat):
    '''
    better_than_baseline(y, yhat):

    computes sum of squared errors for regression coefecients and for baseline and returns boolean value to if the regression model explains errors better than the baseline prediction

    args:
    y: pandas series of target varable
    yhat: pandas series regression predicted variable

    returns boolean value
    '''
    sse = ((yhat - y)**2).sum()
    base_sse = ((y.mean() - y)**2).sum()
    return sse < base_sse


def model_significance(y, yhat, ols_model):
    '''
    model_significance(y, yhat, ols_model)

    explains strength of model by computing amount of explined variance and p value of significance

    args:
    y: pd series target variable
    yhat: pd series regression predicted variable
    ols_model: statsmodels.regression object

    returns:
    dictionary of float explained variance and float p value
    '''

    evs = explained_variance_score(y, yhat)
    f_pval = ols_model.f_pvalue
    return {'exp_var' : evs, 'f_pval' : f_pval}