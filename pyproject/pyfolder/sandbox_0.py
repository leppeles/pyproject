import torch
import numpy as np
import random



x = torch.tensor(10.0,requires_grad = True)

y = x**2 + 2*x + 1

y.backward()

print(x.grad)