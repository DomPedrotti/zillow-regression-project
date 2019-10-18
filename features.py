import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, SelectKBest, f_regression, RFE
from sklearn.linear_model import LassoCV, LinearRegression
import statsmodels.api as sm

import split_scale as ss


def select_kbest_chisquared(x, y, k = 2):


    f_selector = SelectKBest(chi2, k)
    
    f_selector.fit(x,y)
    
    #get feature names
    f_support = f_selector.get_support()
    f_feature = x.loc[:,f_support].columns.tolist()
    return f_feature


def select_kbest_freg(x, y, k = 2):
    '''
    


    '''
    #create selector object for k number of best columns and fit it to data
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(x,y)

    #get names for the k number of best features
    f_support = f_selector.get_support()
    f_feature = x.loc[:,f_support].columns.tolist()
    return f_feature

    
def ols_backward_elimination(x, y):


    cols = x.columns
    while (len(cols)>0):
        x_1 = x[cols]
        model = sm.OLS(y, x_1).fit()
        p = pd.Series(model.pvalues.values[0:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    return cols


def lasso_cv_coef(x, y):
    reg = LassoCV()
    reg.fit(x,y)
    
    coef = pd.Series(reg.coef_, index = x.columns)
    return coef




def optimum_feature_count(x_train, y_train, x_test, y_test):
    number_of_features_list = np.arange(1,len(x_train.columns.tolist())+1)
    high_score = 0
    
    number_of_features = 0
    #score_list = []
    
    for i in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model, number_of_features_list[i])
        x_train_rfe = rfe.fit_transform(x_train, y_train)
        x_test_rfe  = rfe.transform(x_test)
        model.fit(x_train_rfe, y_train)
        score = model.score(x_test_rfe, y_test)
        #score_list.append(score)
        if(score > high_score):
            high_score = score
            number_of_features = number_of_features_list[i]
    return number_of_features

def optimum_feature_names(x_train, y_train, feature_count):
    cols = list(x_train.columns)
    model = LinearRegression()
    
    rfe = RFE(model, feature_count)
    
    x_rfe = rfe.fit_transform(x_train, y_train)
    temp = pd.Series(rfe.support_, index = cols)
    selected_features_rfe = temp[temp==True].index
    
    return selected_features_rfe

def recursive_feature_elimination(features, target, dataframe, train_pct = 0.8):
    cols = features+target

    train, test = ss.split_my_data(dataframe[cols], train_pct = train_pct)
    n = optimum_feature_count(train[features], train[target], test[features], test[target])

    features = optimum_feature_names(train[features], train[target], n)

    train = train[features].join(train[target])
    test = test[features].join(test[target])
    return train, test, features