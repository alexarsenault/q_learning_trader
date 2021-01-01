import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  

def normalize_data(df):
    """Normalize stock prices using the first row of the df."""
    if(len(df.shape) > 1):
        return df/ df.iloc[0,:]
    else:
        return df/df.iloc[0]

def compute_daily_returns(df):
    """Compute daily returns of a portfolio given a df of
    portfolio values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) -1
    daily_returns.iloc[0] = 0
    return daily_returns

def compute_cumulative_returns(df):
    """Computes the cumulative returns of a portfolio given a
    df of daily returns."""
    cumulative_returns = df[-1]/df[0]
    return cumulative_returns

def compute_sharpe_ratio(df, rfr, sf):
    """Computes the sharpe ratio of a portfolio given a df of
    daily returns, risk free rate, and sampling frequency."""
    k = np.sqrt(sf)
    sharpe_ratio = k * (np.mean(df - rfr)/df.std())
    return sharpe_ratio

def error_func(allocs, df):    
    """Computes the negative sharpe ratio of a portfolio given
    allocations and df of stock prices."""

    # only portfolio symbols
    normed = normalize_data(df)
    alloced = normed * allocs
    pos_vals = alloced * 1
    port_vals = pos_vals.sum(axis=1)

    # Compute assessment metrics
    dr = compute_daily_returns(port_vals)       # Daily returns
    sf = 252.0
    rfr = 0.0
    
    sharpe_ratio = compute_sharpe_ratio(dr, rfr, sf) 
    return sharpe_ratio * -1

def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  

    # Read in orders file and sort it 		  	   		     		  		  		    	 		 		   		 		    		  	   		     		  		  		    	 		 		   		 		  
    orders_df = orders_file
    orders_df = orders_df.sort_index()

    # Extract info about orders file and set up data structures
    sd = orders_df.index[0]
    ed = orders_df.index[-1]		  	   		     		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  
    syms = orders_df['Symbol'].unique()
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all['Cash'] = start_val
    prices = prices_all[syms]
    allocs = prices.copy()
    for col in allocs.columns:
        allocs[col].values[:] = 0
    allocs['Cash'] = start_val

    # Iterate through rows and compute transactions
    for row in orders_df.iterrows():
        date = row[0]
        sym = row[1]['Symbol']
        num_shares = abs(row[1]['Shares'])
        sym_price = prices_all.loc[date, sym]

        trade_value = sym_price * num_shares
        trade_cost = commission + (impact * trade_value)
        
        if (row[1]['Shares'] > 0):  # BUY
            allocs.loc[date:,'Cash'] = allocs.loc[date,'Cash'] - trade_value - trade_cost
            allocs.loc[date:,sym] = allocs.loc[date,sym] + num_shares

        elif (row[1]['Shares'] < 0):   # SELL
            allocs.loc[date:,'Cash'] = allocs.loc[date,'Cash'] + trade_value - trade_cost
            allocs.loc[date:,sym] = allocs.loc[date,sym] - num_shares

        else:
            pass

    vals = allocs[syms] * prices[syms]
    vals['Cash'] = allocs['Cash']
    vals['PortVal'] = vals.sum(axis=1)
    
    portvals = vals['PortVal']	  	   		     	
    rv = pd.DataFrame(index=portvals.index, data=portvals.values)  		  	   		     		  		  		    	 		 		   		 		  		  		  		    	 		 		   		 		  
    return rv  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
		  	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		     		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		     		  		  		    	 		 		   		 		  
    pd.set_option('display.max_rows', None)	  	   		     		  		  		    	 		 		   		 		  
    of = "./orders/orders-01.csv"
    
    df = pd.read_csv(of)
    sd = df['Date'][0]		  	   		     		  		  		    	 		 		   		 		  
    ed = df['Date'].iloc[-1]		  	   		     		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   		     		  		  		    	 		 		   		 		  

    #portvals = compute_portvals_copy(df_orders=of, of, start_val=sv) 

    # Process orders  		  	   		     		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		     		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		     		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		     		  		  		    	 		 		   		 		  
    else:  		  	   		     		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Compute assessment metrics
    dr = compute_daily_returns(portvals)        # Daily returns
    adr = dr.mean()                             # Average daily returns
    sddr = dr.std()                             # STD daily returns
    cr = compute_cumulative_returns(portvals)   # Cumulative returns
    #ev = portvals[-1]                          # Ending value
    sr = compute_sharpe_ratio(dr,0,252)         # Sharpe ratio
     		  	   		     		  		  		    	 		 		   		 		  
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [  		  	   		     		  		  		    	 		 		   		 		  
        cr,  		  	   		     		  		  		    	 		 		   		 		  
        adr,  		  	   		     		  		  		    	 		 		   		 		  
        sddr,  		  	   		     		  		  		    	 		 		   		 		  
        sr,  		  	   		     		  		  		    	 		 		   		 		  
    ]
    
    # Compute SPY values for benchmark
    dates = pd.date_range(sd, ed)
    prices_all = get_data([], dates)            # Automatically adds SPY
    prices_SPY = prices_all['SPY']              # Only SPY, for comparison later
    
    normed = normalize_data(prices_SPY)            # Get daily portfolio value

    spy_vals = normed
    dr_spy = compute_daily_returns(spy_vals)       # Daily returns
    adr_spy = dr_spy.mean()                        # Average daily returns
    sddr_spy = dr_spy.std()                        # STD daily returns
    cr_spy = compute_cumulative_returns(spy_vals)  # Cumulative returns
    sr_spy = compute_sharpe_ratio(dr_spy,0,252)    # Sharpe ratio
      		  	   		     		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [  		  	   		     		  		  		    	 		 		   		 		  
        cr_spy,  		  	   		     		  		  		    	 		 		   		 		  
        adr_spy,  		  	   		     		  		  		    	 		 		   		 		  
        sddr_spy,  		  	   		     		  		  		    	 		 		   		 		  
        sr_spy,  		  	   		     		  		  		    	 		 		   		 		  
    ]  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Date Range: {sd} to {ed}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		     		  		  		    	 		 		   		 		  
    print()  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		     		  		  		    	 		 		   		 		  

    df_temp = pd.concat(  		  	   		     		  		  		    	 		 		   		 		  
            [normalize_data(portvals), normalize_data(spy_vals)], keys=["Portfolio", "SPY"], axis=1
        )

    fig,ax=plt.subplots()		 		   		 		  
    df_temp.plot(title='Optimized Portfolio vs. SPY', fontsize=12, ax=ax)  		  	   		     		  		  		    	 		 		   		 		  
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price')
    fig.savefig('plot_1' + '.png')
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
