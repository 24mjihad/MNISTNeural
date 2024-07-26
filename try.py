import numpy as np
#followed the guide by Bot Academy on youtube, line by line 

w_i_h = np.random.uniform(-.5,.5,(4,5)) #4 hidden nodes 5 input nodes

w_h_o = np.random.uniform(-.5,.5,(3,4)) #3 output layer 4 hidden layer

b_i_h = np.zeros((4,1))

b_h_o = np.zeros((3,1))

