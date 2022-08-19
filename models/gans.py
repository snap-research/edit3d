import torch
import torch.nn as nn
import torch.nn.functional as F

def zero_init(m):
    nn.init.constant_(m.weight, 0.)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.)

class Discriminator(nn.Module):
    def __init__(self, d=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(nc, d //2, 4, 2, 1),
            nn.InstanceNorm2d(d // 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d // 2, d, 4, 2, 1),
            nn.InstanceNorm2d(d),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d, d * 2, 4, 2, 1),
            nn.InstanceNorm2d(d * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d * 2, d * 4, 4, 2, 1),
            nn.InstanceNorm2d(d * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d * 4, d * 8, 4, 2, 1),
            nn.InstanceNorm2d(d * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(d * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        outputs = self.net(x)
        return outputs.squeeze()


class Generator(nn.Module):
    def __init__(self, latent_dim, d=64, nc=3):
        super(Generator, self).__init__()

        self.convT1 = nn.ConvTranspose2d(latent_dim, d * 8, 4, 1, 0) # 4x4
        self.bn1 = nn.BatchNorm2d(d * 8)

        self.convT2 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1) #8x8
        self.bn2 = nn.BatchNorm2d(d * 4)

        self.convT3 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1) #16x16
        self.bn3 = nn.BatchNorm2d(d * 2)

        self.convT4 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1) #32x32
        self.bn4 = nn.BatchNorm2d(d)

        self.convT5 = nn.ConvTranspose2d(d, nc, 4, 2, 1) # 64x64

        self.convT5 = nn.ConvTranspose2d(d, d // 2, 4, 2, 1) #64x64
        self.bn5 = nn.BatchNorm2d(d // 2)

        self.convT6 = nn.ConvTranspose2d(d // 2, nc, 4, 2, 1) #128x128

        self.num_layers = 6

    # NOTE remember to freeze the bn when comparing non-linear fine-tuning and linear fine-tuning
    def freeze_bn(self):
        for i in range(self.num_layers - 1):
            bn_i = getattr(self, 'bn'+str(i+1))
            bn_i.eval()
            for p in bn_i.parameters():
                p.requires_grad = False

    def forward(self, x, leaky_relu=True, tanh=True):

        x = self.convT1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, 0.2, inplace=True) if leaky_relu else F.relu(x, inplace=True)

        x = self.convT6(x) # 128x128

        if tanh:
            x = torch.tanh(x)
        return x



