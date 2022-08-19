import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        in_ch = cfg.color_in_ch
        feat_ch = cfg.hidden_ch
        self.color_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 3)
        )
        
    def forward(self, z_color):

        color_out = self.color_net(z_color)
        color_out = torch.sigmoid(color_out) # output is always 0 ~ 1

        return color_out
 

