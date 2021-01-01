""" testproject

This script will run the neccesary scripts to 
generate all of the analysis and plots neccesary 
for the project 8 Strategy Evaluation report.

Author: Alex Arsenault
CS 7646 Machine Learning for Trading

"""

from ManualStrategy import man_strategy_run
from experiment1 import experiment1_run
from experiment2 import experiment2_run


def author():
    return 'aarsenault3'

def main():
    
    man_strategy_run()
    experiment1_run()
    experiment2_run()
    
if __name__ == "__main__":
    main()