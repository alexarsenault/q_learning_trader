"""
Manual Strategy code.

1. Bollinger Band %

2. OBVM

3. PSR

"""

import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd 
import numpy as np 		  	   		     		  		  		    	 		 		   		 		  
import util as ut
import indicators as ind
import marketsimcode as mksm	

import matplotlib.pyplot as plt

class ManualStrategy(object):

    def __init__(  		  	   		     		  		  		    	 		 		   		 		  
        self,
        verbose=False  		  	   		     		  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Constructor method  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.impact = 0.005
        self.commission = 9.95

    def testPolicy(
        self,
        symbol='IBM',  	
        sd=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
        sv=10000,  	 		  	   		     		  		  		    	 		 		   		 		  
    ):
        # Extract neccesary data for indicators
        benchmark_symbol = 'JPM'
        
        dates = pd.date_range(sd, ed)  
        indicators_df = ut.get_data([symbol], dates)                        # Get prices data (automatically adds SPY)
        volume_all = ut.get_data([symbol], dates, colname="Volume")         # Get volume data (automatically adds SPY) 
        indicators_df['Volume'] = volume_all[symbol].values                 # Add volume data to indicators dataframe
        lookback_days = 20
        
        if benchmark_symbol not in indicators_df.columns:
            benchmark_df = ut.get_data([benchmark_symbol], dates)
            indicators_df[benchmark_symbol] = benchmark_df[benchmark_symbol].values
        
        
        # Add BBP, OBVM, PSR
        indicators_df = ind.compute_bbp(indicators_df, symbol, lookback_days)
        indicators_df = ind.compute_obvm(indicators_df, symbol)
        indicators_df = ind.compute_psr(indicators_df, symbol, lookback_days)
        
        """
        Loop through indicators dataframe and determine actions
        based on the value of indicators.
        """
        # Create trades dataframe to track cash, trades, and allocations
        df_trades = pd.DataFrame(index=indicators_df.index)
        df_trades['Trade'] = 0
        
        df_portvals = pd.DataFrame(index=indicators_df.index)
        df_portvals['Cash'] = sv
        df_portvals['Allocations'] = 0
        df_portvals['Price'] = indicators_df[symbol].values
        
        for i in range(lookback_days+1, indicators_df.iloc[lookback_days+1:].shape[0]):
            
            # Indicators
            obvm_ratio_curr = indicators_df.iloc[i]['OBVM_ratio']
            bbp_prev = indicators_df.iloc[i-1]['BBP']
            bbp_curr = indicators_df.iloc[i]['BBP']
            psr = indicators_df.iloc[i]['PSR']
            
            # Current day portfolio and price info
            curr_alloc = df_portvals.iloc[i]['Allocations']
            num_shares_1 = 1000
            trade_val_1 = num_shares_1 * df_portvals['Price'].iloc[i]
            num_shares_2 = 2000
            trade_val_2 = num_shares_2 * df_portvals['Price'].iloc[i]
            
            # Buy signal
            if ( (bbp_curr < 0.2) & (bbp_curr > bbp_prev) & (obvm_ratio_curr > 0.0) & (psr < 1) and curr_alloc < 1000 ):
                
                if (curr_alloc == 0):
                    df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] - trade_val_1 - self.impact*trade_val_1 - self.commission
                    df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] + num_shares_1
                    df_trades['Trade'].iloc[i] = 1000
            
                else:
                    df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] - trade_val_2 - self.impact*trade_val_2 - self.commission
                    df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] + num_shares_2
                    df_trades['Trade'].iloc[i] = 2000
            
            # Sell signal
            elif ( (bbp_curr > 0.8) & (bbp_prev > bbp_curr) & (obvm_ratio_curr < 0.0)  & (psr > 1) and curr_alloc > -1000 ):

                if (curr_alloc == 0):
                    df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] + trade_val_1- self.impact*trade_val_1 - self.commission
                    df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] - num_shares_1 
                    df_trades['Trade'].iloc[i] = -1000
                
                else:
                    df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] + trade_val_2- self.impact*trade_val_2 - self.commission
                    df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] - num_shares_2
                    df_trades['Trade'].iloc[i] = -2000
            
            # Else do nothing
            else:
                pass
        
        df_portvals['Value'] = df_portvals['Cash'] + df_portvals['Allocations'] * df_portvals['Price']

        if (self.verbose == True):    
            df_portvals['Portfolio'] = df_portvals['Value']/df_portvals['Value'].iloc[0]
            df_portvals['Benchmark'] = indicators_df[benchmark_symbol]/indicators_df[benchmark_symbol].iloc[0]   
            
            # Plot normalized returns of manual strategy vs. benchmark with long/short entry lines
            fig,ax=plt.subplots()		 		   		 		  
            ax = df_portvals[['Benchmark', 'Portfolio']].plot(title="Manual Strategy vs. Benchmark", fontsize=12,ax=ax, color=['g', 'r'], grid=True)
            ax.set_xlabel("Date")
            ax.set_ylabel("Normalized Returns")
            ymax = df_portvals['Value'].max()/df_portvals['Value'].iloc[0]
            ymin = 0
            
            if (df_trades[df_trades['Trade']>0].empty==False):
                ax.vlines(x=df_trades[df_trades['Trade']>0].index[0], ymin=ymin, ymax=ymax, colors='blue', linestyles ="dotted", lw=2,  label='Entry Long')
            
            if (df_trades[df_trades['Trade']<0].empty==False):
                ax.vlines(x=df_trades[df_trades['Trade']<0].index[0], ymin=ymin, ymax=ymax, colors='black', linestyles ="dotted", lw=2,  label='Entry Short')
            
            ax.legend()
            fig.savefig('manual_strategy_returns.png', dpi=300)
            
            """            
            indicators_df['JPM'] = indicators_df['JPM']/indicators_df['JPM'].iloc[0]
            indicators_df[['OBVM_ratio','PSR', 'BBP', 'JPM']].plot()
            plt.show()
            
            indicators_df[['OBVM_main','OBVM_sgnl', 'JPM']].plot()
            plt.show()
            
            indicators_df['OBVM_ratio_diff'] = indicators_df['OBVM_ratio'].diff()
            indicators_df[['OBVM_ratio_diff', 'JPM']].plot()
            plt.show()
            
            indicators_df[['OBVM_ratio', 'JPM']].plot()
            plt.show()
            """
                    
        return df_trades
    
    def plot_returns(self, symbol, sd, ed, sv, commission, impact, df_trades, title, filename):
        
        df_trades['Symbol'] = symbol                                                # Add symbol column
        df_trades['Shares'] = df_trades['Trade']
        df_portvals = df_trades.copy()
        df_portvals['Value'] = mksm.compute_portvals(df_trades, sv, commission, impact)
        
        df_portvals['Portfolio'] = df_portvals['Value']/df_portvals['Value'].iloc[0]
        
        # Compute portfolio values of benchmark
        dates = pd.date_range(sd, ed)  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        df_portvals['Benchmark'] = prices_all[symbol]/prices_all[symbol].iloc[0]   
            
        # Plot normalized returns of manual strategy vs. benchmark with long/short entry lines
        fig,ax=plt.subplots()		 		   		 		  
        ax = df_portvals[['Benchmark', 'Portfolio']].plot(title=title, fontsize=12,ax=ax, color=['g', 'r'], grid=True)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Returns")
        ymax = max(df_portvals[['Benchmark','Portfolio']].max())
        ymin = min(df_portvals[['Benchmark','Portfolio']].min())
            
        if (df_trades[df_trades['Trade']>0].empty==False):
            ax.vlines(x=df_trades[df_trades['Trade']>0].index[0], ymin=ymin, ymax=ymax, colors='blue', linestyles ="dotted", lw=3,  label='Entry Long')
            
        if (df_trades[df_trades['Trade']<0].empty==False):
            ax.vlines(x=df_trades[df_trades['Trade']<0].index[0], ymin=ymin, ymax=ymax, colors='black', linestyles ="dotted", lw=3,  label='Entry Short')
            
        ax.legend()
        fig.savefig(filename, dpi=300)
        
        return df_portvals
    
    def compute_daily_returns(self, df):
        """Compute daily returns of a portfolio given a df of
        portfolio values."""
        daily_returns = df.copy()
        daily_returns[1:] = (df[1:] / df[:-1].values) -1
        daily_returns.iloc[0] = 0
        return daily_returns

    def compute_cumulative_returns(self, df):
        """Computes the cumulative returns of a portfolio given a
        df of daily portfolio values."""
        cumulative_returns = df[-1]/df[0]
        return cumulative_returns

    def compute_sharpe_ratio(self, df, rfr, sf):
        """Computes the sharpe ratio of a portfolio given a df of
        daily returns, risk free rate, and sampling frequency."""
        k = np.sqrt(sf)
        sharpe_ratio = k * (np.mean(df - rfr)/df.std())
        return sharpe_ratio
                    
    def compute_metrics(self, portvals):
        
        dr = self.compute_daily_returns(portvals['Value'])         # Daily returns
        adr = dr.mean()                                            # Average daily returns
        sddr = dr.std()                                            # STD daily returns
        cr = self.compute_cumulative_returns(portvals['Value'])    # Cumulative returns
        sr = self.compute_sharpe_ratio(dr, 0, 252)                 # Sharpe ratio

        return [dr, adr, sddr, cr, sr]

def author():
    return 'aarsenault3' 
  
def man_strategy_run():

    # Simulation settings
    symbol = "JPM"
    sd_in_sample=dt.datetime(2008,1,1)
    ed_in_sample=dt.datetime(2009,12,31)
    sd_out_sample=dt.datetime(2010,1,1)
    ed_out_sample=dt.datetime(2011,12,31)
    sv = 100000
    impact = 0.05
    commission = 9.95
    
    # Initialize manual strategy object
    ms = ManualStrategy(verbose=False)
    
    # In sample period
    df_trades_in_sample = ms.testPolicy(symbol = symbol, sd=sd_in_sample, ed=ed_in_sample, sv = sv)   
    
    # Out of sample period
    df_trades_out_sample = ms.testPolicy(symbol = symbol, sd=sd_out_sample, ed=ed_out_sample, sv = sv)   
    
    # Generate in sample and out of sample performance plots
    portvals_in_sample = ms.plot_returns(symbol, sd_in_sample, ed_in_sample, sv, commission, impact, \
        df_trades_in_sample, 'Manual Strategy vs. Benchmark (In Sample)', 'manual_strategy_in_sample_returns.png')
    portvals_out_sample = ms.plot_returns(symbol, sd_out_sample, ed_out_sample, sv, commission, impact, \
        df_trades_out_sample, 'Manual Strategy vs. Benchmark (Out of Sample)' ,'manual_strategy_out_sample_returns.png')
    
    
    # Compute performance metrics of in sample period
    in_sample_port_metrics = ms.compute_metrics(portvals_in_sample)
    
    # Compute performance metrics of in sample period benchmark
    portvals_in_sample_bench_metrics = pd.DataFrame(index=portvals_in_sample.index)
    portvals_in_sample_bench_metrics['Value'] = portvals_in_sample['Benchmark']*1000
    in_sample_bench_metrics = ms.compute_metrics(portvals_in_sample_bench_metrics)
    
    # Compute performance metrics of out of sample period
    out_sample_port_metrics = ms.compute_metrics(portvals_out_sample)
    
    # Compute performance metrics of out of sample period benchmark
    portvals_out_sample_bench_metrics = pd.DataFrame(index=portvals_out_sample.index)
    portvals_out_sample_bench_metrics['Value'] = portvals_out_sample['Benchmark']*1000
    out_sample_bench_metrics = ms.compute_metrics(portvals_out_sample_bench_metrics)
    
    # Print metrics
    """
    print(in_sample_port_metrics[1:])
    print(in_sample_bench_metrics[1:])
    print(out_sample_port_metrics[1:])
    print(out_sample_bench_metrics[1:])
    """

if __name__ == "__main__":
    man_strategy_run()


