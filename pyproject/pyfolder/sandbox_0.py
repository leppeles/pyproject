import torch
import numpy as np
from torch.nn import Linear


# linear equation without torch.nn
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