#3 neurons output here
"""
inputs = [1.2,5.1,2.1]
weights = [3.1,2.1,8.7]
bias =3

output = inputs[0]*weights[0] + inputs[1]*weights[1]+inputs[2]*weights[2] + bias

print(output)

"""

"""
inputs = [1,2,3,2.5]
weights1 = [.2,.8,-.5,1.0]
weights2 = [.5,-.91,.26,-.5]
weights3 = [-.26,-.27,.17,.87]
bias1 =2
bias2 =3
bias3 =0.5

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1]+inputs[2]*weights1[2] +inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1]+inputs[2]*weights2[2] +inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1]+inputs[2]*weights3[2] +inputs[3]*weights3[3] + bias3]

print(output)

"""
'''
#vector * a matrix --> dot product (numpy)
import numpy as np

inputs = [1,2,3,2.5]

weights = [.2,.8,-.5,1]

bias = 2


output = np.dot(weights, inputs) + bias

'''

import numpy as np
'''
inputs = [[1,2,3,2.5],
          [2.0,5,-1,2],
          [-1.5,2.7,3.3,-.8]]

weights = [[.2,.8,-.5,1],
           [.5,-.91,.26,-.5],
           [-.26,-.27,.17,.87]]

bias = [2,3,.5]

#convert to numpy array

#transpose matrix


weights2 = [[.1,-.14,.5],
           [-.5,.12,-.33],
           [-.44,.73,-.13]]

bias2 = [-1,2,-.5]

layer1_output = np.dot(inputs,np.array(weights).T) + bias

layer2_output = np.dot(layer1_output,np.array(weights2).T) + bias2

print(layer2_output)

'''
'''

X = [
    [1,2,3,2.5],
    [2.0,5,-1,2],
    [-1.5,2.7,3.3,-.8]
     ]
#maybe normalize and then scale to get between -1 and 1
np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
#output
#[[ 0.148296   -0.08397602]
# [ 0.14100315 -0.01340469]
# [ 0.20124979 -0.07290616]]

'''

#step function
#Less than zero --> 0
#greater than zero --> 1

#sigmoid --> easier / more accurate (problem with gradient?)

#relu ->simple and very fast


'''
inputs = [0,2,-1,3.3,-2.7,1.1,2.2, -100]

output =[]

for i in inputs:
    if i>0:
        output.append(i)
    elif i <=0:
        output.append(0)

#same thing
for i in inputs:
    output.append(max(0,i))

print(output)
'''

X = [
    [1,2,3,2.5],
    [2.0,5,-1,2],
    [-1.5,2.7,3.3,-.8]
     ]

class Layer_Dense:
    def __init__(self, n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights)+self.biases


class Activaiton_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)