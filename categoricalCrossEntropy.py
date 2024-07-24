import math

softmax_output=[0.7,.1,.2]

#calculate loss on output
target_output = [1,0,0]
#-(log of output)*target output -->sum 
loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])


print(loss)

simple_loss = -(math.log(softmax_output[0]))

print(simple_loss)


