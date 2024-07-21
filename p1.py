#3 neurons output here
"""
inputs = [1.2,5.1,2.1]
weights = [3.1,2.1,8.7]
bias =3

output = inputs[0]*weights[0] + inputs[1]*weights[1]+inputs[2]*weights[2] + bias

print(output)

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


