import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Decoder(nn.Module):
    def __init__(self, cfg, d=64, nc=3):
        super(Decoder, self).__init__()

        latent_dim = cfg.in_ch
        self.imsize = int(np.sqrt(cfg.out_ch))

        self.convT1 = nn.ConvTranspose2d(latent_dim, d * 8, 4, 1, 0) # 4x4
        self.bn1 = nn.BatchNorm2d(d * 8)

        self.convT2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1) #8x8
        self.bn2 = nn.BatchNorm2d(d * 4)

        self.convT3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1) #16x16
        self.bn3 = nn.BatchNorm2d(d * 2)

        self.convT4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1) #32x32
        self.bn4 = nn.BatchNorm2d(d)

        self.convT5 = nn.ConvTranspose2d(d, d // 2, 4, 2, 1) #64x64
        self.bn5 = nn.BatchNorm2d(d // 2)

        self.convT6 = nn.ConvTranspose2d(d // 2, nc, 4, 2, 1) #128x128


    def forward(self, latent_code, leaky_relu=True):

        bz, dim = latent_code.shape
        x = latent_code.view(bz, dim, 1, 1)

        x = self.convT1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT4(x) # 32 x 32
        
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)
        x = self.convT5(x) # 64x64

        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)
        x = self.convT6(x) # 128x128

        logit = x
        x_recon = torch.sigmoid(logit)
        
        return x_recon


