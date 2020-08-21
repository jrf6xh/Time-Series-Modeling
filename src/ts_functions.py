import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
import warnings



# Data Cleaning Functions


def melt_data(df):
    """
    Turn DataFrame from Wide to Long Format
    
    Arguments:
    df -- Pandas DataFrame with dates in Wide format
    
    Return:
    Pandas DataFrame in Long format.
    """
    melted = pd.melt(df, id_vars=['zipcode', 'City', 'State', 'CountyName', 'Metro'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted

def clean_data(df):
    """
    Clean DataFrame.  Rename columns, drop unneeded columns, fill missing Metro values, and transform to Long format.
    
    Arguments:
    df -- Pandas DataFrame in Wide format.
    
    Return:
    Cleaned and melted Pandas DataFrame
    """
    #rename RegionName to 'zipcode'
    df.rename(columns={'RegionName': 'zipcode'}, inplace=True)
    
    #drop unneeded columns
    df.drop(columns=['RegionID', 'RegionType', 'StateName', 'SizeRank'], inplace=True)
    
    #Change to long format
    df_long = melt_data(df)
    
    #Non-metro zipcodes now called 'rural'
    df_long.fillna('rural', inplace=True)
    
    return df_long



#Processing & Viz Functions


def get_codes_data(df, codes):
    """
    Filter DataFrame to show certain zipcodes.
    
    Arguments:
    df -- cleaned Pandas DataFrame in Long format.
    codes -- list of zipcodes to keep in output DataFrame.
    
    Return:
    Pandas DataFrame containing data for the zipcodes contained in 'codes'
    """
    #Filter df to keep data for selected zipcodes
    top_df = df[df['zipcode'].isin(codes)]
    
    #Filter out data from 2020.
    top_df = df.loc[df['time'].dt.year != 2020]
    
    #Set time as index
    top_df.set_index('time', inplace=True)
    
    #Drop unneeded columns
    top_df = top_df[['zipcode', 'value']]
    
    return top_df

def split_data_by_code(top_df, codes):
    """
    Split DataFrame into smaller DataFrames, each containing data for one zipcode.
    
    Arguments:
    top_df -- DataFrame containing data for zipcodes you want to separate.
    codes -- list of zipcodes.  A DataFrame will be created for each item in the list.
    
    Return:
    List of DataFrames where each DataFrame corresponds to data from one zipcode contained in 'codes'
    """
    df_list = []
    for i in range(len(codes)):
        df_list.append(top_df.loc[top_df['zipcode'] == codes[i]].drop('zipcode', axis=1))
        
    return df_list

def ts_train_test_split(df_list, train_percent):
    """
    Split each DataFrame in df_list into train and test sets.
    
    Arguments:
    df_list -- list of DataFrames, each containing data for one zipcode
    train_percent -- float representing the % of data that should be allocated to the training set.
    
    Return:
    train_list -- list of DataFrames, each containing training data for one zipcode
    test_list -- list of DataFrames, each containing test data for one zipcode
    """
    train_list = []
    test_list = []
    
    for i in range(len(df_list)):
        train = df_list[i][:round(df_list[i].shape[0]*train_percent)]
        train_list.append(train)
        
        test = df_list[i][round(df_list[i].shape[0]*train_percent):]
        test_list.append(test)
    
    return train_list, test_list

def plot_trends(df_list, codes):
    """
    Plot line graphs for DataFrames contained in df_list.
    
    Arguments:
    df_list -- list of DataFrames, each containing data for one zipcode
    codes -- list of zipcodes, each corresponding to one element in 'df_list'
    
    Return:
    No return value.  Prints a line graph.
    """
    for i in range(len(codes)):
        df_list[i]['value'].plot(label=codes[i], figsize=(15, 6))
        plt.legend()
        
        
        
# Modeling Functions

def get_difference(df_list):
    """
    Perform 2 orders of differencing on each DataFrame in a list of DataFrames.
    
    Arguments:
    df_list -- list of DataFrames, each containing data for one zipcode
    
    Return:
    list of DataFrames, each containing twice differenced data for one zipcode
    """
    df_diff = []
    for code in df_list:
        df_diff.append(code['value'].diff().diff().dropna())

    return df_diff

def plot_diff_trends(diff_list, codes):
    """
    Plot line graphs for DataFrames contained in diff_list.
    
    Arguments:
    diff_list -- list of DataFrames, each containing differenced data for one zipcode
    codes -- list of zipcodes, each corresponding to one element in 'diff_list'
    
    Return:
    No return value.  Prints a line graph.
    """
    #Graph trends
    for i in range(len(codes)):
        diff_list[i].plot(label=codes[i], figsize=(15, 6))
        plt.legend()
        
def decompose_graphs(diff_list, codes):
    """
    Plot decomposition graphs for DataFrames contained in diff_list.
    
    Arguments:
    diff_list -- list of DataFrames, each containing differenced data for one zipcode
    codes -- list of zipcodes, each corresponding to one element in 'diff_list'
    
    Return:
    No return value.  Prints a decomposition graph for each element in diff_list.
    """
    for i in range(len(codes)):
        decomposition = seasonal_decompose(diff_list[i])
        print(codes[i] + ": ")
        fig = plt.figure()
        fig = decomposition.plot()
        fig.set_size_inches(15, 8)
        
def test_stationarity(timeseries, window):
    """
    Test the stationarity of time series data contained in a DataFrame and print/graph results.
    
    Arguments:
    timeseries -- DataFrame containing time series data
    window -- Integer - the length of time over which to test stationarity
    
    Return:
    N/A.  Prints and graphs results of Dickey Fuller test.
    """

    # Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    
def test_station_of_list(diff_list, window):
    """
    Test the stationarity of multiple DataFrames.
    
    Arguments:
    diff_list -- list of DataFrames containing time series data
    window -- Integer - the length of time over which to test stationarity
    
    Return:
    N/A.  Prints and graphs results of Dickey Fuller test for each DataFrame in diff_list.
    """
    for i in range(len(diff_list)):
        test_stationarity(diff_list[i], window)
        
def plot_all_pacf(diff_list, codes):
    """
    Plot PACF graphs for DataFrames contained in diff_list.
    
    Arguments:
    diff_list -- list of DataFrames, each containing data for one zipcode
    codes -- list of zipcodes, each corresponding to one element in 'diff_list'
    
    Return:
    No return value.  Prints a PACF graph for each element in diff_list.
    """
    for i in range(len(diff_list)):
        plot_pacf(diff_list[i], title= "PACF for Zipcode " + codes[i]);
        
def plot_all_acf(diff_list, codes):
    """
    Plot ACF graphs for DataFrames contained in diff_list.
    
    Arguments:
    diff_list -- list of DataFrames, each containing data for one zipcode
    codes -- list of zipcodes, each corresponding to one element in 'diff_list'
    
    Return:
    No return value.  Prints an ACF graph for each element in diff_list.
    """
    for i in range(len(diff_list)):
        plot_acf(diff_list[i], title="ACF for Zipcode " + codes[i]);
        
def fit_arima(data, model_list, pred_list, ar, diff, ma):
    """
    Fit an ARIMA model on time series data in a DataFrame.  Store model and make predictions.
    
    Arguments:
    data -- DataFrame containing time series data ready for model fitting.
    model_list -- list of models to store the model in.  Can be empty.
    pred_list -- list of predictions to store the predictions in.  Can be empty.
    ar -- Integer - the AutoRegressive term taken as input for the ARIMA model.
    diff -- Integer - the order of differencing term taken as input for the ARIMA model.
    ma -- Integer - the MovingAverage term taken as input for the ARIMA model.
    
    Return:
    model_list -- updated version of the input list containing the new fitted model object.
    pred_list -- updated version of the input list containing a DataFrame of the predictions generated by the model.
    """
    model = ARIMA(data, (ar,diff,ma)).fit()
    model_list.append(model)
    pred_list.append(model.predict(typ='levels'))
    
    return model_list, pred_list
