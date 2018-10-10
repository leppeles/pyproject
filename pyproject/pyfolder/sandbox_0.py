import torch
import numpy as np
from torch.nn import Linear
import matplotlib.pyplot as plt

X = torch.arange(-3,3,0.1).view(-1,1)

f = -3 * X

Y= f + 1.1*torch.randn(X.size())

plt.plot(X.numpy(),f.numpy())
plt.plot(X.numpy(),Y.numpy(),'ro')

w = torch.tensor(-10.0,requires_grad = True)

def forward(x):
    y = w*x
    return y

#calc R^2 tot the given model
def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

learningrate = 0.1

#for epoch in range(5):
LOSS = []
loss = 100
    
while loss > 1.5:
    Yhat = forward(X)
    plt.plot(X.numpy(),Yhat.detach().numpy())
    loss = criterion(Yhat,Y)
    print(Yhat)
    loss.backward()
    w.data = w.data - learningrate * w.grad.data
    w.grad.data.zero_()
    LOSS.append(loss)
    
plt.show()

'''linear equation without torch.nn

w = torch.tensor(20.0,requires_grad = True)
b = torch.tensor(-1.0,requires_grad = True)

def forward(x):
    y = w*x + b
    return y

x = torch.tensor([[1.0],[2.0]])
yhat = forward(x)
print(yhat, "\n"*2)

# linear equation with torch.nn's Linear
torch.manual_seed(10)

model = Linear(in_features=1,out_features=1)
print(list(model.parameters()))

x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
yhat = model(x)
print(yhat)
'''