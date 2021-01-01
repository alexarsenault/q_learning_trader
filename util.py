"""Utility code.
"""

import os
import pandas as pd


def symbol_to_path(symbol, base_dir=None):
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", "./data/")
        return os.path.join(base_dir, "{}.csv".format(str(symbol)))
    
def get_data(symbols, dates, addSPY=True, colname="Adj Close"):
    df = pd.DataFrame(index=dates)
    if addSPY and "SPY" not in symbols:
        symbols = ["SPY"] + list(
            symbols
            )  # handles the case where symbols is np array of 'object'

    for symbol in symbols:
        df_temp = pd.read_csv(
            symbol_to_path(symbol),
            index_col="Date",
            parse_dates=True,
            usecols=["Date", colname],
            na_values=["nan"],
            ) 
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == "SPY":  # drop dates SPY did not trade 
            df = df.dropna(subset=["SPY"])

    return df  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):  		  	   		     		  		  		    	 		 		   		 		  
    import matplotlib.pyplot as plt
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()