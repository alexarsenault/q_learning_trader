In order to produce all of the outputs needed for the analysis, run
the script: testproject.py.

Files contained in this project:

indicators.py
    This file contains functions for computing indicators used in the 
    strategy learner and manual rules trading strategy.

QLearner.py
    This file defines a class used for training and using a
    Q-Learner to learn and use a policy to make decisions.  This class
    is identical to the QLeaner class used in project 7 and is used
    by the StrategyLearner.py class in this project.

marketsim.py
    Contains the code to compute portfolio values given a date range
    and an orders file.

ManualStrategy.py
    This file defines a class to execute the manual strategy for
    this project.  It contains a function called testPolicy, which will
    apply the manual trading strategy based on the input data.  There is
    also a function that can be used to generate plots of the output.
    When run as a script, this file will generate plots required for
    the report showing in sample and out of sample performance of the 
    strategy in the files: 'manual_strategy_in_sample_returns.png' and
    'manual_strategy_out_sample_returns.png'.

StrategyLearner.py
    This files defines a  class needed to train and use the strategy
    learner (which is a QLearner).  Once a StrategyLearner object is
    initialized, the add_evidence method can be used to train is, and
    the testPolicy method can be used to execute it on a particular
    stock and date range.

experiment1.py
    Script that runs experiment 1, which is a comparision of the 
    performance of strategy learner and the manual rules trader.  This 
    script will generate trades from both of the trading strategy 
    objects and generate plots to compare their performance in the
    file:'learner_vs_manual_portolfio.png'.

experiment2.py
    Script that runs experiment 2, which is an evaluation of impact
    penalty on the performance of the strategy learner technique.  The
    script will train and test strategy learner with 4 different impact
    penalty situations and generate a plot of the performances in the 
    file:'impact_effect.png'.

testproject.py
    Script that runs the scripts needed to produce all of the analysis
    and plots for the project 8 report.  Plots generated are:

        -'manual_strategy_in_sample_returns.png'
        -'manual_strategy_out_sample_returns.png'
        -'learner_vs_manual_portfolio.png'
        -'impact_effect.png'