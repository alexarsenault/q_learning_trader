"""
Experiment 1
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

def experiment1_run():

    # Simulation settings
    symbol = "JPM"
    sd=dt.datetime(2008,1,1)
    ed=dt.datetime(2009,12,31)
    sv = 100000
    impact = 0.005
    commission = 9.95

    # Evaluate strategy learner
    learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)   # Constructor
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)                            # Training phase
    df_trades_learner = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)          # Testing phase
    df_trades_learner['Symbol'] = symbol                                                # Add symbol column
    df_trades_learner['Shares'] = df_trades_learner['Trade']
    sl_portvals = ms.compute_portvals(df_trades_learner, sv, commission, impact)        # Compute portvals

    # Evaluate manual strategy
    manual = mst.ManualStrategy(verbose=False)
    df_trades_manual = manual.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv) 
    df_trades_manual['Symbol'] = symbol
    df_trades_manual['Shares'] = df_trades_manual['Trade']
    ms_portvals = ms.compute_portvals(df_trades_manual, sv, commission, impact)

    # Compute portfolio values of benchmark
    dates = pd.date_range(sd, ed)  
    prices_all = ut.get_data([symbol], dates)  # automatically adds SPY

    # Create comparison dataframe
    compare_df = pd.DataFrame(index=df_trades_manual.index)
    compare_df['Learner Portfolio'] = sl_portvals.values
    compare_df['Manual Portfolio'] = ms_portvals.values
    compare_df['Benchmark Portfolio'] = prices_all[symbol].values

    compare_df['Learner Portfolio'] = compare_df['Learner Portfolio']/compare_df['Learner Portfolio'].iloc[0]
    compare_df['Manual Portfolio'] = compare_df['Manual Portfolio']/compare_df['Manual Portfolio'].iloc[0]
    compare_df['Benchmark Portfolio'] = compare_df['Benchmark Portfolio']/compare_df['Benchmark Portfolio'].iloc[0]  

    # Plot learner vs. manual portfolio
    fig,ax=plt.subplots()		 		   		 		  
    ax = compare_df[['Learner Portfolio', 'Manual Portfolio', 'Benchmark Portfolio']].plot(title='Learner vs. Manual vs. Benchmark Performance', fontsize=12,ax=ax, grid=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Returns")
    ax.legend()
    fig.savefig('learner_vs_manual_portolfio.png', dpi=300)

def author():
    return 'aarsenault3' 

if __name__ == "__main__":
    experiment1_run()