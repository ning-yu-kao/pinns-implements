import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import wandb

# Log in and connect to W&B
wandb.login(anonymous="must", key='Your Key')
print("Successfully Logged In!")

# Function for generating GIF
def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

# Build a simple FCN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1) 
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)
        return output

# PDE: f = du/dx - 2du/dt - u 
def f(x,t, model):
    u = model(x,t) 
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    pde = u_x - 2*u_t - u
    return pde

# Function for plotting
def plot_result(title_i:int):
    fig = plt.figure(figsize=(16,6))

    # ==========================
    # First subplot: Prediction
    # ==========================

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    
    x=np.arange(0,2,0.02)
    t=np.arange(0,1,0.02)
    ms_x, ms_t = np.meshgrid(x, t)
    # Just because meshgrid is used, we need to do the following adjustment
    x = np.ravel(ms_x).reshape(-1,1)
    t = np.ravel(ms_t).reshape(-1,1)

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
    pt_u = model(pt_x,pt_t)
    u=pt_u.data.cpu().numpy()
    ms_u = u.reshape(ms_x.shape)

    surf = ax.plot_surface(ms_x,ms_t,ms_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set(title=f'Prediction (epoch= {title_i})')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # ===========================
    # Second subplot: Analytical
    # ===========================
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ms_u = 6*np.exp(-3*ms_x-2*ms_t)

    surf = ax.plot_surface(ms_x,ms_t,ms_u, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                
                

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set(title='Analytical')

    fig.colorbar(surf, shrink=0.5, aspect=5)

# Boundary condition:  u(x,0)=6e^(-3x)
x_bc = np.random.uniform(low=0.0, high=2.0, size=(500,1))
t_bc = np.zeros((500,1))
u_bc = 6*np.exp(-3*x_bc)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
model = model.to(device)
criterion = torch.nn.MSELoss() # mse
optimizer = torch.optim.Adam(model.parameters())
files=[]

wandb.init(
        # Set the project where this run will be logged
        project="PINN-1D-HEAT", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_1", 
        # Track hyperparameters and run metadata
        config={
        "architecture": "PINN",
        "dataset": "1d-heat-pde",
        "epochs": 10000,
        })


iterations = 10000
for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero
    
    # MSE_u: Loss of boundary training data
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = model(pt_x_bc, pt_t_bc) # output of u(x,t)
    mse_u = criterion(net_bc_out, pt_u_bc)
    
    # MSE_f: Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
    all_zeros = np.zeros((500,1))
    
    
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    f_out = f(pt_x_collocation, pt_t_collocation, model) # output of f(x,t)
    mse_f = criterion(f_out, pt_all_zeros)
    
    # Combining the loss functions
    loss = mse_u + 0.1*mse_f  
    # Add a weight for the contribution of loss based on PDE
    # Because we don't want the model relies too much on PDE
    
    loss.backward() # compute gradient
    optimizer.step() # move to next step

    # plot result every 200 epochs
    if (epoch+1) % 200 == 0: 
    
        plot_result(epoch+1)
        
        file = "plots/pinn_%.8i.png"%(epoch+1)
        plt.savefig(file)
        files.append(file)
        
        plt.close("all")

    with torch.autograd.no_grad():
        print(epoch,"Traning Loss:",loss.data)
        wandb.log({"loss": loss.data})

wandb.finish()
save_gif_PIL("pinn.gif", files, fps=20, loop=0)
    
