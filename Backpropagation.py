#forward pass
x= [1,-2,3]
w=[-3,-1,2]
b=1

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

print(xw0,xw1,xw2)

z= xw0+xw0+xw2+b
print(z)

y= max(z,0) #relu
print(y)

dvalue=1 # derivative from next layer

relu_dz = (1 if z>0 else 0) #derivative of relu with respect to z

drelu_dz = dvalue *(1 if z>0 else 0)

print(drelu_dz)

#partial derivative of the multiplication, the chain rule
dsum_dxw0 = 1
drelu_dxw0 = drelu_dz*dsum_dxw0
print(drelu_dxw0)


dsum_dxw1 = 1
drelu_dxw1 = drelu_dz*dsum_dxw1
print(drelu_dxw1)

dsum_dxw2 = 1
drelu_dxw2 = drelu_dz*dsum_dxw2
print(drelu_dxw1)

dsum_db = 1
drelu_db = drelu_dz*dsum_db


print(drelu_dxw0,drelu_dxw1,drelu_dxw2,drelu_db)

dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0*dmul_dx0

#simple version derivative of RELU with respect to first input -- with chain rule

drelu_dx0 = dvalue * (1 if z>0 else 0) * w[0]


dx = [drelu_dx0, drelu_dxw1, drelu_dxw2] # gradients on input
dw = [drelu_dxw0, drelu_dxw1, drelu_dxw2] # gradients on weights
db = drelu_db # gradient on bias..just 1 here.

