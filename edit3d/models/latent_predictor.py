import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from models.resnet import resnet18


class MLP(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(MLP, self).__init__()
        self.fc0 = nn.Linear(in_ch, hidden_ch)
        self.fc1 = nn.Linear(hidden_ch, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, out_ch)

    def forward(self, z):
        y = F.leaky_relu(self.fc0(z), 0.2, inplace=True)
        y = F.leaky_relu(self.fc1(y), 0.2, inplace=True)
        logit = self.fc2(y)
        return logit


class LatentNN(nn.Module):
    def __init__(self, latent_dim):
        super(LatentNN, self).__init__()
        self.imagenn = resnet18(in_channel=6, low_dim=latent_dim)
        # assume the outptu of imagenn is same as the dim of latent code
        self.mlp = MLP(latent_dim * 2, latent_dim * 2, latent_dim)

    def forward(self, image, init_latent):

        im_latent = self.imagenn(image)
        joint_latent = torch.cat([im_latent, init_latent], dim=-1)
        delta_latent = self.mlp(joint_latent)
        return delta_latent + joint_latent
