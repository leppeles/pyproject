import torch
import matplotlib.pyplot as plt
from torch import nn,optim
import plotter

#mini-batch gradient descent

torch.manual_seed(1)

from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1, 1)
        self.f=1*self.x-1
        self.y=self.f+1.1*torch.randn(self.x.size())
        self.len=self.x.shape[0]
    def __getitem__(self,index):    
            
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len
    
dataset = Data()

plt.plot(dataset.x.numpy(),dataset.y.numpy(),'rx',label='y')
plt.plot(dataset.x.numpy(),dataset.f.numpy(),label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
    def forward(self,x):
        yhat=self.linear(x)
        return yhat
    
criterion = nn.MSELoss()

model=linear_regression(1,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

trainloader=DataLoader(dataset=dataset,batch_size=5)

model.state_dict()['linear.weight'][0]=-15
model.state_dict()['linear.bias'][0]=-10

get_surface=plotter.plot_error_surfaces(15,13,dataset.x,dataset.y,30,go=False)

epochs=5

for epoch in range(epochs):
    for x,y in trainloader:
        #make a prediction 
        yhat=model(x)
        #calculate the loss
        loss=criterion(yhat,y)
        #for plotting 
        get_surface.get_stuff(model,loss.tolist())
        #clear gradient 
        optimizer.zero_grad()
        #Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        #the step function on an Optimizer makes an update to its parameters
        optimizer.step()

    get_surface.plot_ps()

