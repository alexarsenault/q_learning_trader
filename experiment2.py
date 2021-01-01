"""
Experiment 2
"""

import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import random  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  
import numpy as np		  	   		     		  		  		    	 		 		   		 		  
import util as ut
import QLearner as ql
import StrategyLearner as sl
import indicators as ind	
import time

import matplotlib.pyplot as plt  
import marketsimcode as ms
import ManualStrategy as mst

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

def experiment2_run():
    # Simulation settings
    symbol = "JPM"
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    impact_arr = [0.0, 0.005, 0.025, 0.050]
    commission = 9.95
    sl_portvals = []
    num_trades = []

    for impact in impact_arr:
        # Evaluate strategy learner
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)   # Constructor
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)                            # Training phase
        df_trades_learner = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)          # Testing phase
        df_trades_learner['Symbol'] = symbol                                                # Add symbol column
        df_trades_learner['Shares'] = df_trades_learner['Trade']
        sl_portvals.append(ms.compute_portvals(df_trades_learner, sv, commission, impact))  # Compute portvals
        num_trades.append(df_trades_learner.astype(bool).sum(axis=0)[0])

    # Create comparison dataframe for portvals
    compare_df = pd.DataFrame(index=sl_portvals[0].index)

    for i in range(len(sl_portvals)):
        compare_df[str(impact_arr[i])] = sl_portvals[i].values
        compare_df[str(impact_arr[i])] = compare_df[str(impact_arr[i])]/compare_df[str(impact_arr[i])].iloc[0]

    # Plot all normalized performances
    fig,ax=plt.subplots()		 		   		 		  
    ax = compare_df.plot(title='Effect of Impact on QLearner Returns', fontsize=12,ax=ax, grid=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Returns")
    ax.legend()
    fig.savefig('impact_effect.png', dpi=300)


    # Create comparison list for metrics
    compare_df_metrics = []

    for i in range(len(sl_portvals)):
        impact = impact_arr[i]
        dr = compute_daily_returns(compare_df[str(impact)])         # Daily returns
        adr = dr.mean()                                             # Average daily returns
        sddr = dr.std()                                             # STD daily returns
        cr = compute_cumulative_returns(compare_df[str(impact)])    # Cumulative returns
        sr = compute_sharpe_ratio(dr, 0, 252)                       # Sharpe ratio

        compare_df_metrics.append([impact, dr, adr, sddr, cr, sr])
        
    print("")

def author():
    return 'aarsenault3' 

if __name__ == "__main__":
    experiment2_run()
