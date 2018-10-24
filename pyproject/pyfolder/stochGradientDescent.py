#Created on 2018. okt. 10.

import torch
import matplotlib.pyplot as plt
import numpy as np
import plotter

from mpl_toolkits import mplot3d

torch.manual_seed(1)

X=torch.arange(-3,3,0.1).view(-1, 1)
w=torch.tensor(-10.0,requires_grad=True)
f=-3*X+1
Y=f+1.1*torch.randn(X.size())

plt.plot(X.numpy(),Y.numpy(),'rx',label='y')
plt.plot(X.numpy(),f.numpy(),label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

def forward(x):
    return w*x+b

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

get_surface=plotter.plot_error_surfaces(15,13,X,Y,30)

epochs=10
LOSS1=[]
w=torch.tensor(-15.0,requires_grad=True)
b=torch.tensor(-10.0,requires_grad=True)

for epoch in range(epochs):
    #SGD is an approximation of out true total loss/cost, in this line of code we calculate our true loss/cost and store it
    Yhat=forward(X)
    #store the loss 
    LOSS1.append(criterion(Yhat,Y).tolist())
    #each iteration of the loop corresponds to one iteration of SGD
    for x,y in zip(X,Y):
        #make a prediction 
        yhat=forward(x)
        #calculate the loss 
        loss=criterion(yhat,y)
        #update state of plotting object not Pytroch 
        get_surface.get_stuff(w.data.tolist(),b.data.tolist(),loss.tolist())
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        #updata parameters slope
        w.data=w.data-lr*w.grad.data
        b.data=b.data-lr*b.grad.data
        #clear gradients 
        w.grad.data.zero_()
        b.grad.data.zero_()
    #plot surface and data space after each epoch    
    get_surface.plot_ps()
    

