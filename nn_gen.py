import torch
import torch.nn as nn
import torch.nn.functional as func

class VAE(nn.Module):
    '''
    Neural network class.
    Architecture:
        One convolution layer conv1 (plus relu, max pool, then batch norm)
        Two fully-connected layers fc1 and fc2.
        Two nonlinear activation functions relu and sigmoid.
    '''

    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(1,1,2)
        self.batch = nn.BatchNorm2d(1)
        self.fc1= nn.Linear(169, 100)
        self.fc21= nn.Linear(100, 20)
        self.fc22= nn.Linear(100, 20)
        self.fc3= nn.Linear(20, 100)
        self.fc4= nn.Linear(100, 196)

    # Feedforward function
    def encoder(self, x):
        conv1 = self.batch(func.relu(self.conv1(x)))
        h1 = conv1.reshape(*conv1.shape[:2],-1) # flatten last two dimensions
        h2 = func.relu(self.fc1(h1))
        mu = self.fc21(h2)
        logvar = self.fc22(h2)
        return mu, logvar

    def reparamaterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decoder(self, z):
        h3 = func.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        out_size = h4.size()
        y = h4.reshape([out_size[0], 1, 14, 14])
        return y

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparamaterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        x = x/255
        BCE = func.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # # Backpropagation function
    # def backprop(self, data, loss, epoch, optimizer):
    #     self.train()
    #     inputs= torch.from_numpy(data.x_train)
    #     targets= torch.from_numpy(data.y_train)
    #     outputs= self(inputs)
    #     obj_val= loss(self.forward(inputs), targets)
    #     optimizer.zero_grad()
    #     obj_val.backward()
    #     optimizer.step()
    #     return obj_val.item()

    # # Test function. Avoids calculation of gradients.
    # def test(self, data, loss, epoch):
    #     self.eval()
    #     with torch.no_grad():
    #         inputs= torch.from_numpy(data.x_test)
    #         targets= torch.from_numpy(data.y_test)
    #         outputs= self(inputs)
    #         cross_val= loss(self.forward(inputs), targets)
    #     return cross_val.item()
