import random as rand
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  

class QLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """
    Q-learner object.
    """

    def __init__(  		  	   		     		  		  		    	 		 		   		 		  
        self,  		  	   		     		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		     		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		     		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		     		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		     		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		     		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		     		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		     		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		     		  		  		    	 		 		   		 		  
    ):  		  	   		     		  		  		    	 		 		   		 		  
        """
        Constructor
        """
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = np.zeros(shape=(num_states,num_actions))
        self.Tc = np.ones(shape=(self.num_states, self.num_actions, self.num_states))
        self.R = np.ones(shape=(self.num_states, self.num_actions))
        self.exp_tuple = []

	  	   		     		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		     		  		  		    	 		 		   		 		  
        """
        Update the state without updating the Q-table		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        self.s = s  		  	   		     		  		  		    	 		 		   		 		  
        action = self.q_table[s,:].argmax()
        self.a = action

        if self.verbose:  		  	   		     		  		  		    	 		 		   		 		  
            print(f"s = {s}, a = {action}")  		  	   		     		  		  		    	 		 		   		 		  
        return action  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		     		  		  		    	 		 		   		 		  
        """
        Update the Q table and return an action

        Update function:
        Q[s,a] = Q[s,a] +  alpha*( r[s,a] + gamma*( Q[s',:].max() - Q[s,a] ) )
        """

        # Update Q table
        self.q_table[self.s,self.a] = self.q_table[self.s,self.a] +  \
            self.alpha*( r + self.gamma*( self.q_table[s_prime,:].max() - \
            self.q_table[self.s,self.a] ) )

        # Decide on next action
        if (np.random.random() <= self.rar):                    # Pick a random action   		  	   		     		  		  		    	 		 		   		 		  
            action = rand.randint(0, self.num_actions - 1)
        else:                                                   # Pick action with highest Q value
            action = self.q_table[s_prime,:].argmax()

        self.exp_tuple.append((self.s, self.a, s_prime, r))

        # Dyna component
        if (self.dyna > 0):
            
            self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha*r

            exp_tuple_len = len(self.exp_tuple)
            random_tuple = np.random.randint(exp_tuple_len, size=self.dyna)

            for i in range(self.dyna):

                # Pick random experience tuples that we have encountered
                temp_tuple = self.exp_tuple[random_tuple[i]]
                s_rand = temp_tuple[0]
                a_rand = temp_tuple[1]
                s_rand_prime = temp_tuple[2]
                r_rand = temp_tuple[3]

                # Update Q table
                self.q_table[s_rand, a_rand] = self.q_table[s_rand, a_rand] +  \
                    self.alpha*( r_rand + self.gamma*( self.q_table[s_rand_prime,:].max() - \
                    self.q_table[s_rand, a_rand] ) )

        # Update state variables
        self.rar = self.rar * self.radr                         # Update random action rate
        self.s = s_prime                                        # Update current state
        self.a = action                                         # Update new action

        return action
