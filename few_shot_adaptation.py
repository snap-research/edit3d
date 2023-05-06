import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import yaml
from shutil import copy2
import sys
import importlib
from functools import partial
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.optim as optim


def render_all(trainer, feat_shape, feat_color):
    shape_img = trainer.render_express(feat_shape, feat_color)
    sketch_img = trainer.render_sketch(feat_shape)
    sketch_img = np.uint8(cv2.resize(sketch_img, shape_img.shape[:2]) * 255)
    color_img = trainer.render_color2d(feat_color, feat_shape)
    color_img = np.uint8(cv2.resize(color_img, shape_img.shape[:2]) * 255)
    return color_img, shape_img, sketch_img


def render_batch(trainer, feat_shape, feat_color, path):
    out = []
    for i in range(min(16, feat_shape.size(0))):
        rgb, shape, sketch = render_all(
            trainer, feat_shape[i : i + 1], feat_color[i : i + 1]
        )
        out.append(cv2.hconcat([rgb, shape, sketch]))
    out = cv2.vconcat(out)
    cv2.imwrite(path, out)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class mineGAN(nn.Module):
    def __init__(self, vad, code_type, nz=128):
        super(mineGAN, self).__init__()
        self.vad = vad
        self.shape_miner = nn.Sequential(
            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),
            nn.Linear(nz, nz),
        )
        self.color_miner = nn.Sequential(
            nn.Linear(nz, nz),
            nn.BatchNorm1d(nz),
            nn.ReLU(),
            nn.Linear(nz, nz),
        )
        if code_type == "shape":
            self.code_type = ["shape"]
        elif code_type == "color":
            self.code_type = ["color"]
        elif code_type == "both":
            self.code_type = ["shape", "color"]
        else:
            self.code_type = []

    def freeze_vad(self):
        for p in self.vad.parameters():
            p.requires_grad = False
        for m in self.vad.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, feat_shape, feat_color, skip_miner=False):
        if ("shape" in self.code_type) and (not skip_miner):
            feat_shape = self.shape_miner(feat_shape)
        if ("color" in self.code_type) and (not skip_miner):
            feat_color = self.color_miner(feat_color)
        color_img = self.vad.forward_color2d_grad(feat_color, feat_shape)
        return color_img, feat_shape, feat_color


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=4):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


# -----------------------------------------------------------
# -------------------MineGAN for VAD-------------------------
# -----------------------------------------------------------

parser = argparse.ArgumentParser(description="Few-shot shape generation")
parser.add_argument("config", type=str, help="The configuration file.")
parser.add_argument(
    "--pretrained", default=None, type=str, help="pretrained MM-VADs checkpoint"
)
parser.add_argument("--outf", default=None, type=str)
parser.add_argument("--code", default="shape", choices=["shape", "color", "both"])
parser.add_argument("--dataset", default="armchair", type=str)
parser.add_argument("--niter", default=100, type=int)
parser.add_argument("--bz", default=32, type=int, help="batch size")
parser.add_argument("--nimgs", default=-1, type=int, help="max number of images")
parser.add_argument("--kld", default=0.02, type=float, help="weight on KLD loss")
parser.add_argument("--beta", default=0.5, type=float, help="margin on KLD loss")
parser.add_argument("--lr", default=0.0002, type=float, help="weight on KLD loss")
parser.add_argument("--mode", default="train", choices=["train", "test"])
args = parser.parse_args()
batch_size = args.bz

os.makedirs(args.outf, exist_ok=True)

TRAIN_STAGE = args.mode == "train"
TEST_STAGE = args.mode == "test"


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# parse config file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
cfg = dict2namespace(config)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
trainer_lib = importlib.import_module(cfg.trainer.type)
trainer = trainer_lib.Trainer(cfg, args, device)

# Define the miner MLP network for mineGAN
netM = mineGAN(trainer, args.code).to(device)
netM.freeze_vad()
latent_dim = 128  # both shape and color has 128 dim


# --------------------------------------------------------------
# ----------------------- Train MM-VADs ------------------------
# --------------------------------------------------------------

if TRAIN_STAGE:

    # load pretrained MM-VADs
    trainer.resume_demo(args.pretrained)
    idx2sid = {}
    for k, v in trainer.sid2idx.items():
        idx2sid[v] = k

    # dataset
    dataroot = args.dataset
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        ),
    )
    if args.nimgs != -1:
        dataset = torch.utils.data.Subset(dataset, range(args.nimgs))
    print("Number of images: %d" % len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    # define discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # optimizer
    if args.code == "shape":
        optimizerM = optim.Adam(
            netM.shape_miner.parameters(), lr=args.lr, betas=(0.99, 0.999)
        )
    elif args.code == "color":
        optimizerM = optim.Adam(
            netM.color_miner.parameters(), lr=args.lr, betas=(0.99, 0.999)
        )
    elif args.code == "both":
        optimizerM = optim.Adam(
            list(netM.color_miner.parameters()) + list(netM.shape_miner.parameters()),
            lr=args.lr,
            betas=(0.99, 0.999),
        )
    else:
        print("Unknown!")
        exit()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.99, 0.999))

    # train the model
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0

    fixed_shape = torch.randn(batch_size, latent_dim, device=device)
    fixed_color = torch.randn(batch_size, latent_dim, device=device)
    fake, feat_shape, feat_color = netM(fixed_shape, fixed_color)

    has_save = False
    for epoch in range(args.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=real_cpu.dtype, device=device
            )

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            shape_noise = torch.randn(batch_size, latent_dim, device=device)
            color_noise = torch.randn(batch_size, latent_dim, device=device)
            fake, feat_shape, feat_color = netM(shape_noise, color_noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_M_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update M network: maximize log(D(M(z)))
            ###########################
            netM.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            # loss_kld = torch.mean(0.5 * torch.mean(feat_shape**2, dim=-1)) # KLD loss
            # loss_kld = args.kld * (torch.clamp(loss_kld, args.beta, None) - args.beta)
            loss_kld = torch.tensor(0, device=device)
            errM = criterion(output, label)
            errMiner = errM + loss_kld
            errMiner.backward()
            D_M_z2 = output.mean().item()
            optimizerM.step()

            print(
                "Stage 1: [%d/%d][%d/%d] Loss_D: %.4f Loss_M: %.4f Loss_KLD: %.4f D(x): %.4f D(M(z)): %.4f / %.4f"
                % (
                    epoch,
                    args.niter,
                    i,
                    len(dataloader),
                    errD.item(),
                    errM.item(),
                    loss_kld.item(),
                    D_x,
                    D_M_z1,
                    D_M_z2,
                )
            )
        if (epoch + 1 >= 50) and ((epoch + 1) % 10 == 0):
            if not has_save:
                for real_i in range(real_cpu.size(0)):
                    vutils.save_image(
                        real_cpu[real_i],
                        "%s/real_%d.png" % (args.outf, real_i),
                        normalize=True,
                    )
                    has_save = True
            fake, feat_shape, feat_color = netM(fixed_shape, fixed_color)
            render_batch(
                netM.vad,
                feat_shape,
                feat_color,
                "%s/fake_all_epoch_%03d.png" % (args.outf, epoch),
            )
            vutils.save_image(
                fake.detach(),
                "%s/fake_2d_epoch_%03d.png" % (args.outf, epoch),
                normalize=True,
            )
            torch.save(netM.state_dict(), "%s/netM_epoch_%d.pth" % (args.outf, epoch))


# --------------------------------------------------------------
# ----------------------- Test MM-VADs -------------------------
# --------------------------------------------------------------

if TEST_STAGE:

    ckpt = torch.load(args.pretrained, map_location=device)
    netM.load_state_dict(ckpt, strict=False)

    for i in range(10):
        shape = torch.randn(batch_size, latent_dim, device=device)
        color = torch.randn(batch_size, latent_dim, device=device)
        fake, feat_shape, feat_color = netM(shape, color)
        render_batch(netM.vad, feat_shape, feat_color, "%s" % (args.outf), i)
