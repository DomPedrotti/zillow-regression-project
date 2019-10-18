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