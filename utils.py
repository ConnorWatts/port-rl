from networks.critic import Critic
from networks.actor import Actor
import torch.nn.functional as F
import torch

#### getters

def get_actor(config):
    # TO DO:
    # pass in device
    # Actor(envs).to(device)
    return Actor(config)

def get_critic(config):
    # TO DO:
    return Critic(config)

def get_loss(config):
    # TO DO:
    return F.mse_loss()

def get_actor_noise(config: dict, device):

    # TO DO: allow variation

    dim = len(config['stocks'])
    dist = config['actor_noise_dist']

    if dist =='gaussian':
        loc = torch.zeros(dim).to(device)
        scale = torch.ones(dim).to(device)
        normal = torch.distributions.normal.Normal(loc, scale)
        return torch.distributions.independent.Independent(normal,1)
    elif dist =='uniform':
        return torch.distributions.Uniform(torch.zeros(dim).to(device),torch.ones(dim).to(device))

