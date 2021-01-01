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
     		  		  		    	 		 		   		 		  
class StrategyLearner(object):
    
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        
    def discretize_indicators(self, df, ind1, ind2, ind3):
        df[ind1] = pd.qcut(df[ind1],10, labels=False)
        df[ind2] = pd.qcut(df[ind2],10, labels=False)
        df[ind3] = pd.qcut(df[ind3],10, labels=False)
        return df
    
    def discretize_pos(self, pos):
        return int(pos[0]*300 + pos[1]*100 + pos[2]*10 + pos[3])   

    def compute_reward(self, portvals, i):
        return portvals['Allocations'].iloc[i] * ((portvals.iloc[i]['Price'] / portvals.iloc[i-1]['Price']) - 1)
         		  		    	 		 		   		 		  
    # This method should create a QLearner, and train it for trading  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
        ):	  	   		     		  		

        # Extract neccesary data for indicators
        start_time = time.time()
        end_time = 0

        benchmark_symbol = 'JPM'
        dates = pd.date_range(sd, ed)  
        indicators_df = ut.get_data([symbol], dates)                        # Get prices data (automatically adds SPY)
        volume_all = ut.get_data([symbol], dates, colname="Volume")         # Get volume data (automatically adds SPY) 
        indicators_df['Volume'] = volume_all[symbol].values                 # Add volume data to indicators dataframe
        lookback_days = 20
        
        if benchmark_symbol not in indicators_df.columns:
            benchmark_df = ut.get_data([benchmark_symbol], dates)
            indicators_df[benchmark_symbol] = benchmark_df[benchmark_symbol].values
        
        
        indicators_df = ind.compute_bbp(indicators_df, symbol, lookback_days)                   # Add BBP
        indicators_df = ind.compute_obvm(indicators_df, symbol)                                 # Add OBVM
        indicators_df = ind.compute_psr(indicators_df, symbol, lookback_days)                   # Add PSR
        indicators_df = ind.compute_momentum(indicators_df, symbol, lookback_days)              # Add Momentum
        indicators_df = self.discretize_indicators(indicators_df, 'BBP', 'OBVM_ratio', 'PSR')   # Discretize indicator values
    		  		  		    	 		 		   		 		  
        # Learning code begins here  		  	   		     		  		  		    	 		 		   		 		  
        self.learner = ql.QLearner( 
            num_states=3000,
            num_actions=3,
            alpha=0.2,
            gamma=0.9,
            rar=0.98,
            radr=0.999,
            dyna=200,  		  	   		     		  		  		    	 		 		   		 		  
            verbose=False,
            )
        
        """
        Loop through indicators dataframe and determine actions
        based on the value of indicators.
        """
        epochs = 30
        r = 0
        cum_rewards = [0.00001]
        num_shares_1 = 1000
        num_shares_2 = 2000

        if (self.verbose==True):
            print("Setup time: " + str(time.time() - start_time) + ".")
        
        for epoch in range(1, epochs + 1):   
                       
            # Create trades dataframe to track cash, trades, and allocations
            #df_trades = pd.DataFrame(index=indicators_df.index)
            #df_trades['Trade'] = 0
                
            df_portvals = pd.DataFrame(index=indicators_df.index)
            df_portvals['Cash'] = sv
            df_portvals['Allocations'] = 0.00001
            df_portvals['Price'] = indicators_df[symbol].values
            action = 2
        
            for i in range(lookback_days+1, indicators_df.iloc[lookback_days+1:].shape[0]):

                # Compute current day indicators
                obvm_ratio_curr = indicators_df.iloc[i]['OBVM_ratio']
                bbp_curr = indicators_df.iloc[i]['BBP']
                psr = indicators_df.iloc[i]['PSR']
                
                # Current day portfolio and price info
                curr_alloc = df_portvals.iloc[i]['Allocations']
                
                # Compute reward                
                r = self.compute_reward(df_portvals, i)
                
                # Current state and initial action
                if (i == lookback_days+1):
                    state = self.discretize_pos([action, obvm_ratio_curr, bbp_curr, psr])
                    action = self.learner.querysetstate(state)  # set the state and get first action  		
                else:
                    state = self.discretize_pos([action, obvm_ratio_curr, bbp_curr, psr])
                    action = self.learner.query(state,r)  # set the state and get action 	    
                
                # Execute action produced by Q learner    
                if (action == 0 and curr_alloc < 1000):     # BUY
                    if (curr_alloc == 0 ):
                        df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] - (num_shares_1 * df_portvals['Price'].iloc[i]) - self.impact*(num_shares_1 * df_portvals['Price'].iloc[i]) - self.commission
                        df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] + num_shares_1 
                        #df_trades['Trade'].iloc[i] = 1000
            
                    else:
                        df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] - (num_shares_2 * df_portvals['Price'].iloc[i]) - self.impact*(num_shares_2 * df_portvals['Price'].iloc[i]) - self.commission
                        df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] + num_shares_2
                        #df_trades['Trade'].iloc[i] = 2000
                        
                elif (action == 1 and curr_alloc > -1000):  # SELL

                    if (curr_alloc == 0 ):
                        df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] + (num_shares_1 * df_portvals['Price'].iloc[i]) - self.impact*(num_shares_1 * df_portvals['Price'].iloc[i]) - self.commission
                        df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] - num_shares_1
                        #df_trades['Trade'].iloc[i] = -1000
                    
                    else:
                        df_portvals['Cash'].iloc[i:] = df_portvals['Cash'].iloc[i] + (num_shares_2 * df_portvals['Price'].iloc[i]) - self.impact*(num_shares_2 * df_portvals['Price'].iloc[i]) - self.commission
                        df_portvals['Allocations'].iloc[i:] = df_portvals['Allocations'].iloc[i] - num_shares_2
                        #df_trades['Trade'].iloc[i] = -2000
                        
                else:               # HOLD
                    pass
            
            # Final reward value (cumulative returns)    
            r = df_portvals.iloc[-1]['Cash'] + df_portvals.iloc[-1]['Allocations'] * df_portvals.iloc[-1]['Price']
            cum_rewards.append(r)

            if ((epoch >= 15) & (cum_rewards[epoch]/cum_rewards[epoch-1] < 1.05) & (cum_rewards[epoch]/cum_rewards[epoch-1] > 0.95) or (epoch >= 20) ):
                if (self.verbose==True):
                    end_time = time.time()
                    print("Total time taken: " + str(end_time-start_time) + ".")
                    plt.plot(cum_rewards)
                    plt.show()
                break
            
        if (self.verbose==True):
            end_time = time.time()
            plt.plot(cum_rewards)
            plt.show()
            print("Total time taken: " + str(end_time-start_time) + ".")

    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
        ):  		  	   		     		  		  		    	 		 		   		 		  

        # Set up data to work with
        start_time = time.time()
        end_time = 0
        dates = pd.date_range(sd, ed)  
        indicators_df = ut.get_data([symbol], dates)                        # Get prices data (automatically adds SPY)
        volume_all = ut.get_data([symbol], dates, colname="Volume")         # Get volume data (automatically adds SPY) 
        indicators_df['Volume'] = volume_all[symbol].values                 # Add volume data to indicators dataframe
        lookback_days = 20
        benchmark_symbol = 'JPM'

        if(self.verbose==True):
            if benchmark_symbol not in indicators_df.columns:
                benchmark_df = ut.get_data([benchmark_symbol], dates)
                indicators_df[benchmark_symbol] = benchmark_df[benchmark_symbol].values
        
        # Create indicators dataframe and discretize appropriately
        indicators_df = ind.compute_bbp(indicators_df, symbol, lookback_days)                   # Add BBP
        indicators_df = ind.compute_obvm(indicators_df, symbol)                                 # Add OBVM
        indicators_df = ind.compute_psr(indicators_df, symbol, lookback_days)                   # Add PSR
        indicators_df = self.discretize_indicators(indicators_df, 'BBP', 'OBVM_ratio', 'PSR')   # Discretize indicator values

        if(self.verbose==True):
            print("Setup test policy time: "  + str(time.time() - start_time) + ".")

        # Create trades dataframe to track cash, trades, and allocations
        df_trades = pd.DataFrame(index=indicators_df.index)
        df_trades['Trade'] = 0
                
        df_portvals = pd.DataFrame(index=indicators_df.index)
        df_portvals['Cash'] = sv
        df_portvals['Allocations'] = 0
        df_portvals['Price'] = indicators_df[symbol].values
        
        list_trades = []
        list_cash = np.ones(shape=indicators_df.shape[0])*sv
        list_allocs = np.zeros(shape=indicators_df.shape[0])*sv
        
        action = 2
        
        for i,row in indicators_df.iloc[lookback_days+1:].iterrows():
                
            # Current day portfolio and price info
            curr_alloc = df_portvals.loc[i]['Allocations']

            # Current state and action
            state = self.discretize_pos([action, row['OBVM_ratio'], row['BBP'], row['PSR']])
            action = self.learner.querysetstate(state)  # set the state and get first action  		
                 
            # Execute action produced by Q learner    
            if (action == 0 and curr_alloc < 1000):     # BUY
                if (curr_alloc == 0 ):
                    df_portvals['Cash'].loc[i:] = df_portvals['Cash'].loc[i] - (1000 * df_portvals['Price'].loc[i]) - self.impact*(1000 * df_portvals['Price'].loc[i]) - self.commission
                    df_portvals['Allocations'].loc[i:] = curr_alloc + 1000
                    list_trades.append(1000)
            
                else:
                    df_portvals['Cash'].loc[i:] = df_portvals['Cash'].loc[i] - (2000 * df_portvals['Price'].loc[i]) - self.impact*(2000 * df_portvals['Price'].loc[i]) - self.commission
                    df_portvals['Allocations'].loc[i:] = curr_alloc + 2000
                    list_trades.append(2000)
                        
            elif (action == 1 and curr_alloc > -1000):  # SELL
                if (curr_alloc == 0 ):
                    df_portvals['Cash'].loc[i:] = df_portvals['Cash'].loc[i] + (1000 * df_portvals['Price'].loc[i]) - self.impact*(1000 * df_portvals['Price'].loc[i]) - self.commission
                    df_portvals['Allocations'].loc[i:] = curr_alloc - 1000
                    list_trades.append(-1000)
                   
                else:
                    df_portvals['Cash'].loc[i:] = df_portvals['Cash'].loc[i] + (2000 * df_portvals['Price'].loc[i]) - self.impact*(2000 * df_portvals['Price'].loc[i]) - self.commission
                    df_portvals['Allocations'].loc[i:] = curr_alloc - 2000
                    list_trades.append(-2000)    
                        
            else:               # HOLD
                list_trades.append(0)

        df_trades['Trade'].iloc[lookback_days+1:] = list_trades
        
        if (self.verbose == True):
            print("Iteration test policy time: "  + str(time.time() - start_time) + ".")
            
        return df_trades  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  

def main():
    
    learner = sl.StrategyLearner(verbose = True, impact = 0.005, commission=9.95) # constructor
    learner.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 12, 31, 0, 0), sv=100000) # training phase
    df_trades = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 12, 31, 0, 0), sv = 100000)  # testing phase
    df_trades = learner.testPolicy(symbol = "IBM", sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 12, 31, 0, 0), sv = 100000)   # testing phase
    
def author():
    return 'aarsenault3' 
  	   	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    main()	  	   		     		  		  		    	 		 		   		 		  
