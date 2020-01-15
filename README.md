# Local-Options
Methods for learning local options, Q-learning given local options and experiments

agents.py contains agent classes that implement a policy defined by a dictionary with states
as keys and arrays of action probabilities as entries. 

algorithms.py contains implementations of Q-learning, SARSA, Td(0) 
that work with vectorized rewards (action selection is handled by scalarization in Q-learning). It also contains algorithms to learn option
models, either by solving a local MDP for different rewards and solving the resulting system of equations, or by an approach based on learning 
the transition model. 

environments.py contains a generic MDP class, that implements a transition and reward model. There are also functions to randomly generate an MDP and the robot MDP:

![alt text](https://github.com/flodorner/Local-Options/blob/master/images/Robot.jpg "robot MDP") 

wrapper.py contains wrappers that use a base MDP and turn it into a local MDP on parts of the state space. 

Test.py contains
