import torch
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal

device = 'cpu'

class DynamicsEncoder(nn.Module):
    """ Dynamics parameters encoding network: (params) -> (latent code) """
    def __init__(self, param_dim, latent_dim, hidden_dim=32, hidden_activation=F.relu, output_activation=F.tanh, num_hidden_layers=2, lr=1e-3, gamma=0.99):
        super(DynamicsEncoder, self).__init__()
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self._param_dim = param_dim
        self.latent_dim = latent_dim
        self.num_hidden_layers = num_hidden_layers

        self.input_layer =  nn.Linear(self._param_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        self.hidden_layers = nn.ModuleList(self.hidden_layers)  # Have to wrap the list layers with nn.ModuleList to coorectly make those parameters tracked by nn.module! Otherwise those params will not be saved!
        self.output_layer =  nn.Linear(hidden_dim, latent_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x=self.hidden_activation(self.input_layer(x))
        for hl in self.hidden_layers:
            x=self.hidden_activation(hl(x))
        x=self.output_layer(x)
        if self.output_activation is not None:
            x=self.output_activation(x)
        return x

class EmbeddingDynamicsNetwork(nn.Module):
    """ Common class for dyanmics prediction network with dynamics embedding as input: (s,a, alpha) -> s' """
    def __init__(self, s_dim, a_dim, latent_dim, hidden_dim=32, hidden_activation=F.relu, output_activation=F.tanh, num_hidden_layers=2, lr=1e-3, gamma=0.99):
        super(EmbeddingDynamicsNetwork, self).__init__()
        
        in_size = s_dim+a_dim+latent_dim
        out_size = s_dim

        self.weights1 =  nn.Parameter(torch.randn(in_size, hidden_dim))
        self.bias1 = nn.Parameter(torch.randn(hidden_dim))
        self.weights2 =  nn.Parameter(torch.randn(hidden_dim, out_size))
        self.bias2 = nn.Parameter(torch.randn(out_size))
        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = self.relu(x @ self.weights1 + self.bias1)
        y = x @ self.weights2 + self.bias2
        return y

class DynamicsParamsOptimizer():
    """ 
    Dynamics parameters optimization model (gradient-based) based on a trained 
    forward dynamics prediction network: (s, a, learnable_params) -> s_ with real-world data. 
    """
    def __init__(self, s_dim, a_dim, param_dim, latent_dim, hidden_dim=32, hidden_activation=F.relu, output_activation=None, num_hidden_layers=4, lr=1e-2, gamma=0.99):
        self.dynamics_model = EmbeddingDynamicsNetwork(s_dim, a_dim, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr, gamma).to(device)
        self.dynamics_encoder = DynamicsEncoder(param_dim, latent_dim, hidden_dim, hidden_activation, output_activation, num_hidden_layers, lr, gamma).to(device)
        self.optimizer = torch.optim.Adam(list(self.dynamics_model.parameters()) + list(self.dynamics_encoder.parameters()), lr=lr)

        self.loss = nn.MSELoss()

    def forward(self, x, theta):
        """ s,a concat with param (learnable) -> s_ """

        alpha = self.dynamics_encoder(theta)
        y_  = self.dynamics_model(torch.cat((x, alpha), axis=-1))
        
        return y_

    def update(self, data, epoch=200, model_save_path=None):
        (x, theta, y) = data
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(device)
        if not isinstance(theta, torch.Tensor):
            theta = torch.Tensor(theta).to(device)        
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y).to(device)

        for ep in range(epoch):
            y_ = self.forward(x, theta)
            self.optimizer.zero_grad()
            loss = self.loss(y_, y)
            loss.backward()
            self.optimizer.step()
            if ep%100==0:
                print('epoch: {}, loss: {}'.format(ep, loss.item()))
                torch.save(self.dynamics_model.state_dict(), model_save_path+'dynamics_model')
                torch.save(self.dynamics_encoder.state_dict(), model_save_path+'dynamics_encoder')


class EmbeddingFit(PyroModule):
    def __init__(self, latent_dim, dynamics_model):
        super().__init__()
        self.alpha = PyroSample(dist.Normal(0., 1.).expand([latent_dim]).to_event(1))
        # self.weights1 = copy.deepcopy(dynamics_model.weights1.cpu())
        # self.bias1 = copy.deepcopy(dynamics_model.bias1.cpu())
        # self.weights2 = copy.deepcopy(dynamics_model.weights2.cpu())
        # self.bias2 = copy.deepcopy(dynamics_model.bias2.cpu())
        # self.bias2 = torch.randn(11, requires_grad=False)
        self.dynamics_model = dynamics_model

        # self.sigma = pyro.sample("sigma", dist.Uniform(0., 1.).expand([1]).to_event(1))
        self.sigma = pyro.sample("sigma", dist.LogNormal(0., 0.01).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, output=None):
        batch_size = x.shape[0]
        input = torch.cat((x, self.alpha.repeat([batch_size, 1])), axis=-1)
        # x = self.relu(input @ self.weights1 + self.bias1)
        # mu = x @ self.weights2 + self.bias2
        mu = self.dynamics_model(input)
        with pyro.plate("instances", batch_size):
            return pyro.sample("obs", dist.Normal(mu, self.sigma).to_event(1),  # TODO whether 0.01 or self.sigma, self.sigma does not seem to be updated
                               obs=output)

        # return mu