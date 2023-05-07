import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.fc0 = nn.Linear(cfg.in_ch, cfg.hidden_ch)
        self.fc1 = nn.Linear(cfg.hidden_ch, cfg.hidden_ch)
        self.fc2 = nn.Linear(cfg.hidden_ch, cfg.out_ch)
        self.imsize = int(np.sqrt(cfg.out_ch))

    def forward(self, z):
        y = F.leaky_relu(self.fc0(z), 0.2, inplace=True)
        y = F.leaky_relu(self.fc1(y), 0.2, inplace=True)
        logit = self.fc2(y)
        logit = logit.view(-1, 1, self.imsize, self.imsize)
        x_recon = F.sigmoid(logit)
        return logit, x_recon
