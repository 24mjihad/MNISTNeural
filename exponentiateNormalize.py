import numpy as np
import nnfs
nnfs.init()

layer_outputs = [[4.8,1.21,2.385],
                 [8.9,-1.81,.2],
                 [1.41,1.051,0.026]]


exp_values = np.exp(layer_outputs)
print(exp_values)

norm_value = exp_values/np.sum(exp_values)

print(norm_value)
print(sum(norm_value))