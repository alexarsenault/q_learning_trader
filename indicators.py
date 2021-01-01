"""
Technical indicators code

Author: Alex Arsenault

"""

import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data

def compute_sma(df, symbol, lookback_period):
    """ Function to compute the simple moving average over a given
    lookback period. Return value is a new dataframe.
    """
    #df_sma = df.copy()
    #df_sma['SMA'] = df[symbol].cumsum()
    #df_sma['SMA'].values[lookback_period:] = (df_sma['SMA'].values[lookback_period:] - df_sma['SMA'].values[:-lookback_period])/lookback_period
    #df_sma['SMA'].values[0:lookback_period] = np.nan
    df['SMA'] = df[symbol].cumsum()
    df['SMA'].values[lookback_period:] = (df['SMA'].values[lookback_period:] - df['SMA'].values[:-lookback_period])/lookback_period
    df['SMA'].values[0:lookback_period] = np.nan

    return df

def compute_bbp(df, symbol, lookback_period):
    """ Function to compute the bollinger band over a given lookback
    period. If SMA is not already part of the dataframe, compute_sma()
    will be called. Return value is a new dataframe.
    """
    
    if 'SMA' not in df.columns:
        df_bbp = compute_sma(df, symbol, lookback_period)
    else:
        df_bbp = df.copy()
           
    df_bbp['Std'] = df_bbp[symbol].rolling(window=lookback_period).std()                 # Rolling std  
    df_bbp['BBU'] = df_bbp['SMA'] + (2*df_bbp['Std'])                                    # Upper band
    df_bbp['BBL'] = df_bbp['SMA'] - (2*df_bbp['Std'])                                    # Lower band
    df_bbp['BBP'] = (df_bbp[symbol] - df_bbp['BBL']) / (df_bbp['BBU'] - df_bbp['BBL'])   # BB percentage

    return df_bbp

def compute_psr(df, symbol, lookback_period):
    """ Function to compute the price to SMA ratio, which should have
    more value as an indicator than price or SMA alone. If SMA is not
    already computed, compute_sma() is called with the input lookback
    period. Return value is a new dataframe.
    """
    if 'SMA' not in df.columns:
        df_psr = compute_sma(df, symbol, lookback_period)
    else:
        df_psr = df.copy()
        
    df_psr['PSR'] = df_psr[symbol]/df_psr['SMA']
   
    return df_psr

def compute_momentum(df, symbol, lookback_period):
    """ Computes momentum over fixed number of days. Return value is a
    new dataframe.
    """
    df_momentum = df.copy()
    df_momentum['Momentum'] = df_momentum[symbol]
    df_momentum['Momentum'].values[lookback_period:] = df[symbol].values[lookback_period:]/df[symbol].values[:-lookback_period] - 1
    df_momentum['Momentum'].values[0:lookback_period] = np.nan
    
    return df_momentum

def compute_obvm(df, symbol):
    
    if 'Volume' not in df.columns:      # For now must have Volume already computed
        pass
    else:
        df_obv = df.copy() 

    df_obv['OBV_vect'] = df_obv[symbol].diff()
    df_obv['OBV_updown'] = ((df_obv['OBV_vect'] > 0).astype(int))*2 -1
    df_obv['OBV'] = (df['Volume'].iloc[1:] * df_obv['OBV_updown']).cumsum()
    
    df_obv['OBV'] = df_obv['OBV'] + abs(df_obv['OBV'].min()) 
    
    df_obv['OBVM_main'] = df_obv['OBV'].ewm(span=7, adjust=False).mean()     
    df_obv['OBVM_sgnl'] = df_obv['OBV'].ewm(span=10, adjust=False).mean()   #changed num of days for signal line
    
    df_obv['OBVM_ratio'] = df_obv['OBVM_main']/df_obv['OBVM_sgnl'] - 1
    
    return df_obv    

def compute_volatility(df, symbol, num_days):
    """ Computes volatility over fixed number of days.
    """
    df_volatility = df.copy()
    df_volatility['Volatility'] = np.nan
    df_volatility['Volatility'] = df[symbol].rolling(window=num_days).std()
    
    return df_volatility

def plot_data(df, xlabel, ylabel, title, num):
    """ Standard plot function for dataframe values.
    """
    fig,ax=plt.subplots()		 		   		 		  
    df.plot(title=title, fontsize=12, ax=ax)  		  	   		     		  		  		    	 		 		   		 		  
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig('plot_' + num + '.png')
    
def plot_bb_data(df, xlabel, ylabel, title, num):
    """ Plot function for bollinger band values.
    """
    fig,ax=plt.subplots(2)
    
    df['UL Threshold'] = 1
    df['LL Threshold'] = 0
    		 		   		 		  
    df.plot(y=['BBU', 'BBL', 'SMA', 'JPM'], title=title, fontsize=10, ax=ax[0])  		  	   		     		  		  		    	 		 		   		 		  
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    
    df.plot(y=['BBP','UL Threshold','LL Threshold'], fontsize=10, ax=ax[1])  		  	   		     		  		  		    	 		 		   		 		  
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('BB %')
    
    plt.tight_layout()
    
    fig.savefig('plot_' + num + '.png')
    
def plot_obvm_data(df, xlabel, ylabel, title, num):
    """ Plot function for OBVM.
    """
    fig,ax=plt.subplots(2)		 		   		 		  
    df.plot(y=['OBV', 'OBVM_main', 'OBVM_sgnl'], title=title, fontsize=10, ax=ax[0])  		  	   		     		  		  		    	 		 		   		 		  
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    
    df.plot(y=['OBVM_ratio'], fontsize=10, ax=ax[1])  		  	   		     		  		  		    	 		 		   		 		  
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('OBVM Ratio')
    
    plt.tight_layout()
    
    fig.savefig('plot_' + num + '.png')

def author():
    return 'aarsenault3'

def main():
    # Set up variables to request data
    symbol = 'JPM'
    starting_cash = 100000
    sd=dt.datetime(2008, 1, 1)  		  	   		     		  		  		    	 		 		   		 		  
    ed=dt.datetime(2008, 12, 31)
    
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)                   # Get price data (automatically adds SPY)  		  	   		     		  		  		    	 		 		   		 		  
    volume_all = get_data([symbol], dates, colname="Volume") # Get volume data (automatically adds SPY)  
    
    prices = prices_all[[symbol]]                            # only portfolio symbols
    prices_SPY = prices_all[['SPY']]                     # only SPY, for comparison later
    
    prices['Volume'] = volume_all[[symbol]].values           # Add volume for OBV computation
    prices_SPY['Volume'] = volume_all['SPY'].values
       
    # Compute momentum
    momentum_days = 20
    df_momentum = compute_momentum(prices, symbol, momentum_days)
    # Plot and save figure
    plot_data(df_momentum['Momentum'], 'Date', '20 Day Momentum', 'JPM Momentum', '1')
    
    # Compute volatility 20 day
    volatility_days = 20
    df_volatility = compute_volatility(prices, symbol, volatility_days)
    plot_data(df_volatility['Volatility'], 'Date', 'Volatility', 'JPM Volatility (20 Day)', '2')
    
    # Compute volatility 50 day
    volatility_days = 50
    df_volatility = compute_volatility(prices, symbol, volatility_days)
    plot_data(df_volatility['Volatility'], 'Date', 'Volatility', 'JPM Volatility (50 Day)', '5')
        
    # Compute simple moving average
    lookback_period = 20
    df_sma = compute_sma(prices, symbol, lookback_period)
    plot_data(df_sma[['JPM', 'SMA']], 'Date', '20 Day SMA', 'JPM SMA', '3')
    
    # Compute bollinger band %
    df_bbp = compute_bbp(prices, symbol, lookback_period)
    plot_bb_data(df_bbp, 'Date', '', 'Bollinger Band 20 Day', '4')
    
    # Compute price/sma ratio
    df_psr = compute_psr(prices, symbol, lookback_period)    
    plot_data(df_psr['PSR'], 'Date', 'Price/SMA', 'JPM PSR', '6')
        
    # Compute OBV
    df_obv = compute_obvm(prices, symbol)
    plot_obvm_data(df_obv, 'Date', 'OBV', 'OBVM', '7')
    
if __name__ == "__main__":
    main()