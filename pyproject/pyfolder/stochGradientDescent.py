#Created on 2018. okt. 10.

import torch
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits import mplot3d

class plot_error_surfaces(object):
    def __init__(self,w_range, b_range,X,Y,n_samples=30,go=True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z=np.zeros((30,30))
        count1=0
        self.y=Y.numpy()
        self.x=X.numpy()
        for w1,b1 in zip(w,b):
            count2=0
            for w2,b2 in zip(w1,b1):
                Z[count1,count2]=np.mean((self.y-w2*self.x+b2)**2)
                count2 +=1
    
            count1 +=1
        self.Z=Z
        self.w=w
        self.b=b
        self.W=[]
        self.B=[]
        self.LOSS=[]
        self.n=0
        if go==True:
            plt.figure()
            plt.figure(figsize=(7.5,5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
            plt.title('Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    def get_stuff(self,W,B,loss):
        self.n=self.n+1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
        
    def final_plot(self): 
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c='r', marker='x',s=200,alpha=1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x,self.y,'ro',label="training points")
        plt.plot(self.x,self.W[-1]*self.x+self.B[-1],label="estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: '+str(self.n))
        plt.legend()
        plt.show()
        plt.subplot(122)
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W,self.B,c='r', marker='x')
        plt.title('Loss Surface Contour Iteration'+str(self.n) )
        plt.xlabel('w')
        plt.ylabel('b')
        plt.legend()

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

get_surface=plot_error_surfaces(15,13,X,Y,30)

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
    

