# PyTorch
import time

import clip
import cv2
import lpips
import torch

import edit3d
from edit3d.models.clip_loss import CLIPLoss
from edit3d.trainers.trainer_multimodal import Trainer as CrossModalTrainer


class Trainer(CrossModalTrainer):
    def set_percept_loss(self, device):
        self.lpips_loss = lpips.LPIPS(net="vgg").to(device)

    def set_clip_loss(self):
        self.clip_loss = CLIPLoss(image_size=128)

    def _get_render_sdfs(self, zz_shape, zz_color):
        def sdf_fun(p):  # p: N, 3
            N = p.size(0)
            p = p.unsqueeze(0)

            z_shape = zz_shape
            if len(z_shape.shape) == 2:
                z_shape = z_shape.unsqueeze(1)
            z_shape = z_shape.expand(-1, N, -1)
            inp = torch.cat([z_shape, p], dim=-1)
            dists, shape_feats = self.deepsdf_net(inp)  # [1 N 1]
            dists = dists.reshape(-1, 1)

            z_color = zz_color
            if len(z_color.shape) == 2:
                z_color = z_color.unsqueeze(1)
            z_color = z_color.expand(-1, N, -1)
            inp = torch.cat([z_color, shape_feats, p], dim=-1)
            color3d = self.colorsdf_net(inp)  # [1 N 1]
            color3d = color3d.reshape(-1, 3)
            return {"dists": dists, "color3d": color3d}

        return sdf_fun, None

    def get_known_latent(self, idx):
        num_known_shapes = len(self.sid2idx)
        if idx is None:
            return num_known_shapes
        data_indices = torch.tensor([idx], dtype=torch.long, device=self.device)
        # shape
        latent_codes_coarse_shape, latent_codes_fine_shape, kld = self._b_idx2latent(
            self.latent_embeddings_shape, data_indices, num_augment_pts=1
        )  # [64 128]
        loss_kld_shape = torch.mean(0.5 * torch.mean(latent_codes_coarse_shape ** 2, dim=-1))
        self.stats_loss_kld_shape = loss_kld_shape.item()
        # color
        latent_codes_coarse_color, latent_codes_fine_color, kld = self._b_idx2latent(
            self.latent_embeddings_color, data_indices, num_augment_pts=1
        )  # [64 128]
        loss_kld_color = torch.mean(0.5 * torch.mean(latent_codes_coarse_color ** 2, dim=-1))
        self.stats_loss_kld_color = loss_kld_color.item()
        return latent_codes_coarse_shape, latent_codes_coarse_color

    def manip_fun(self, x, target, mask, feat, gamma=0.02, beta=0.5, alphas=[1.0, 1.0, 0.0]):
        loss_kld = torch.mean(0.5 * torch.mean(feat ** 2, dim=-1))
        loss_kld = gamma * (torch.clamp(loss_kld, beta, None) - beta)
        loss_man = alphas[0] * torch.mean(torch.abs(x - target) * mask.to(edit3d.device)) + alphas[1] * loss_kld
        return loss_man, loss_kld

    def sample_latent_gaussian(self, num_pts):
        latent_codes_coarse_color = self.latent_embeddings_color.random_sample_gaussian(num_pts)
        latent_codes_coarse_shape = self.latent_embeddings_shape.random_sample_gaussian(num_pts)
        return (
            latent_codes_coarse_shape["latent_code"],
            latent_codes_coarse_color["latent_code"],
        )

    def step_manip_sketch(
        self,
        feat_shape,
        target,
        mask=None,
        gamma=0.02,
        beta=0.5,
        alphas=[1.0, 1.0, 0.0],
        epoch=1001,
    ):
        if mask == None:
            mask = torch.ones_like(target)
        latent_codes_coarse = feat_shape.to(self.device).clone().detach().requires_grad_(True)
        optim, lrscheduler = self._get_optim([latent_codes_coarse], self.cfg.manip.optim)
        target = target.to(self.device)
        mask = mask.to(self.device)
        latent_codes = []
        for param in self.imgen_net.parameters():
            param.requires_grad = False
        latent_codes.append(latent_codes_coarse.detach().clone())
        for i in range(epoch):
            optim.zero_grad()
            _, sketch = self.imgen_net(latent_codes_coarse)
            loss_manip, loss_kld = self.manip_fun(
                sketch,
                target,
                mask,
                latent_codes_coarse,
                gamma=gamma,
                beta=beta,
                alphas=alphas,
            )
            loss_manip.backward()
            if i % 100 == 0:
                print(i, loss_manip.item(), loss_kld.item())
            optim.step()
            latent_codes.append(latent_codes_coarse.detach().clone())
        return latent_codes, loss_manip

    def step_manip_color(self, feat_shape, feat_color, target, mask, alphas=[1.0, 1.0, 0.0]):
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(True)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(False)
        optim, lrscheduler = self._get_optim([latent_codes_color], self.cfg.manip.optim_rgb)
        latent_codes = []
        for param in self.colorgen_net.parameters():
            param.requires_grad = False
        target = target.to(self.device)
        mask = mask.to(self.device)
        for i in range(1000):
            optim.zero_grad()
            color_2d = self.forward_color2d_grad(latent_codes_color, latent_codes_shape)
            loss_manip = self.manip_fun(color_2d, target, mask, latent_codes_color, alphas=alphas)
            loss_manip.backward()
            optim.step()
            latent_codes.append(latent_codes_color.detach().clone())
        return latent_codes

    def recon_fun(self, x, target, mask, feat_shape, feat_color, gamma=0.02, beta=0.5):
        loss_kld = torch.mean(0.5 * torch.mean(feat_shape ** 2, dim=-1))
        loss_kld = gamma * (torch.clamp(loss_kld, beta, None) - beta)
        loss_kld2 = torch.mean(0.5 * torch.mean(feat_color ** 2, dim=-1))
        loss_kld2 = gamma * (torch.clamp(loss_kld2, beta, None) - beta)
        loss_man = torch.mean(torch.abs(x - target) * mask.to(edit3d.device)) + loss_kld + loss_kld2
        return loss_man, loss_kld, loss_kld2

    def step_recon_rgb(
        self,
        feat_shape,
        feat_color,
        target,
        mask=None,
        epoch=1001,
        gamma=0.02,
        beta=0.5,
    ):
        if mask == None:
            mask = torch.ones_like(target)
        mask = mask.to(self.device)
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(True)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(True)
        optim, lrscheduler = self._get_optim([latent_codes_shape, latent_codes_color], self.cfg.manip.optim_rgb)
        latent_codes = []
        for param in self.colorgen_net.parameters():
            param.requires_grad = False
        target = target.to(self.device)
        for i in range(epoch):
            optim.zero_grad()
            color_2d = self.forward_color2d_grad(latent_codes_color, latent_codes_shape)
            loss_recon, loss_kld, loss_kld2 = self.recon_fun(
                color_2d,
                target,
                mask,
                latent_codes_shape,
                latent_codes_color,
                gamma=gamma,
                beta=beta,
            )
            loss_recon.backward()
            if i % 100 == 0:
                print(i, loss_recon.item(), loss_kld.item(), loss_kld2.item())
            optim.step()
            latent_codes.append(
                (
                    latent_codes_shape.detach().clone(),
                    latent_codes_color.detach().clone(),
                )
            )
        return latent_codes, loss_recon

    def step_edit_rgb(
        self,
        feat_shape,
        feat_color,
        target,
        mask=None,
        epoch=1001,
        gamma=0.02,
        beta=0.5,
    ):
        if mask == None:
            mask = torch.ones_like(target)
        mask = mask.to(self.device)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(False)
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(True)
        # self.cfg.manip.optim_rgb.lr = 0.0001
        optim, lrscheduler = self._get_optim([latent_codes_color], self.cfg.manip.optim_rgb)
        latent_codes = []
        for param in self.colorgen_net.parameters():
            param.requires_grad = False
        target = target.to(self.device)
        for i in range(epoch):
            optim.zero_grad()
            color_2d = self.forward_color2d_grad(latent_codes_color, latent_codes_shape)
            loss_recon, loss_kld, loss_kld2 = self.recon_fun(
                color_2d,
                target,
                mask,
                latent_codes_shape,
                latent_codes_color,
                gamma=gamma,
                beta=beta,
            )
            loss_recon.backward()
            if i % 100 == 0:
                print(i, loss_recon.item(), loss_kld.item(), loss_kld2.item())
            optim.step()
            latent_codes.append(
                (
                    latent_codes_shape.detach().clone(),
                    latent_codes_color.detach().clone(),
                )
            )
        return latent_codes, loss_recon

    def step_edit_sketch(self, feat_shape, feat_color, target, epoch=1001, gamma=0.02, beta=0.5):
        mask = torch.ones_like(target)
        mask = mask.to(self.device)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(True)
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(False)
        optim, lrscheduler = self._get_optim([latent_codes_shape], self.cfg.manip.optim_rgb)
        latent_codes = []
        for param in self.imgen_net.parameters():
            param.requires_grad = False
        target = target.to(self.device)

        for i in range(epoch):
            optim.zero_grad()
            _, sketch = self.imgen_net(latent_codes_coarse)
            loss_manip, loss_kld = self.manip_fun(
                sketch,
                target,
                mask,
                latent_codes_coarse,
                gamma=gamma,
                beta=beta,
                alphas=alphas,
            )
            loss_manip.backward()
            if i % 100 == 0:
                print(i, loss_manip.item(), loss_kld.item())
            optim.step()
            latent_codes.append(latent_codes_coarse.detach().clone())

        return latent_codes, loss_recon

    def step_edit_sketch(self, feat_shape, target, mask=None, gamma=0.02, beta=0.5, epoch=1001):
        if mask == None:
            mask = torch.ones_like(target)
        alphas = [1.0, 1.0, 0.0]
        latent_codes_coarse = feat_shape.to(self.device).clone().detach().requires_grad_(True)
        optim, lrscheduler = self._get_optim([latent_codes_coarse], self.cfg.manip.optim)
        target = target.to(self.device)
        mask = mask.to(self.device)
        latent_codes = []
        for param in self.imgen_net.parameters():
            param.requires_grad = False
        latent_codes.append(latent_codes_coarse.detach().clone())
        for i in range(epoch):
            optim.zero_grad()
            _, sketch = self.imgen_net(latent_codes_coarse)
            loss_manip, loss_kld = self.manip_fun(
                sketch,
                target,
                mask,
                latent_codes_coarse,
                gamma=gamma,
                beta=beta,
                alphas=alphas,
            )
            loss_manip.backward()
            if i % 100 == 0:
                print(i, loss_manip.item(), loss_kld.item())
            optim.step()
            latent_codes.append(latent_codes_coarse.detach().clone())
        return latent_codes, loss_manip

    def step_clip_color(self, feat_shape, feat_color, text, gamma=0.02, beta=0.5):
        text_input = torch.cat([clip.tokenize(text)]).to(self.device)
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(True)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(False)
        self.cfg.manip.optim.lr = 10
        self.cfg.manip.optim.lr_scheduler.initial = 10
        self.cfg.manip.optim.lr_scheduler.interval = 10
        self.cfg.manip.optim.lr_scheduler.factor = 0.5

        def step_lr(step, optim, lrscheduler):
            lr = lrscheduler(step)
            for g in optim.param_groups:
                g["lr"] = lr
            return lr

        optim, lrscheduler = self._get_optim([latent_codes_color], self.cfg.manip.optim)
        latent_codes = []
        for param in self.colorgen_net.parameters():
            param.requires_grad = False
        since = time.time()
        for i in range(200):
            optim.zero_grad()
            color_2d = self.forward_color2d_grad(latent_codes_color, latent_codes_shape)
            loss_kld = torch.mean(0.5 * torch.mean(latent_codes_color ** 2, dim=-1))
            loss_kld = gamma * (torch.clamp(loss_kld, beta, None) - beta)
            loss_manip = self.clip_loss(color_2d, text_input) + loss_kld
            loss_manip.backward()
            optim.step()
            lr = step_lr(i, optim, lrscheduler)
            latent_codes.append(latent_codes_color.detach().clone())
            print(i, loss_manip.item(), loss_kld.item(), lr)
        print("optimization takes %f seconds" % (time.time() - since))
        return latent_codes

    def step_clip_shape(self, feat_shape, feat_color, text, gamma=0.02, beta=0.5):
        text_input = torch.cat([clip.tokenize(text)]).to(self.device)
        latent_codes_color = feat_color.to(self.device).clone().detach().requires_grad_(False)
        latent_codes_shape = feat_shape.to(self.device).clone().detach().requires_grad_(True)
        self.cfg.manip.optim.lr_scheduler.initial = self.cfg.manip.optim.lr = 5
        self.cfg.manip.optim.lr_scheduler.interval = 20
        self.cfg.manip.optim.lr_scheduler.factor = 0.5

        def step_lr(step, optim, lrscheduler):
            lr = lrscheduler(step)
            for g in optim.param_groups:
                g["lr"] = lr
            return lr

        optim, lrscheduler = self._get_optim([latent_codes_shape], self.cfg.manip.optim)
        latent_codes = []
        for param in self.colorgen_net.parameters():
            param.requires_grad = False
        since = time.time()
        MSE = torch.nn.MSELoss()
        for i in range(100):
            optim.zero_grad()
            color_2d = self.forward_color2d_grad(latent_codes_color, latent_codes_shape)
            loss_kld = torch.mean(0.5 * torch.mean(latent_codes_shape ** 2, dim=-1))
            loss_kld = gamma * (torch.clamp(loss_kld, beta, None) - beta)
            loss_l2 = MSE(latent_codes_shape, feat_shape.to(self.device).clone().detach())
            loss_manip = self.clip_loss(color_2d, text_input) + loss_kld + 0.01 * loss_l2
            loss_manip.backward()
            optim.step()
            lr = step_lr(i, optim, lrscheduler)
            latent_codes.append(latent_codes_shape.detach().clone())
            print(i, loss_manip.item(), loss_kld.item(), loss_l2.item(), lr)
        print("optimization takes %f seconds" % (time.time() - since))
        return latent_codes

    # render 3D shapes
    def render_express(self, feat_shape, feat_color=None, resolution=512):
        if feat_color == None:
            colorize = False
            _, feat_color = self.get_known_latent(0)
        else:
            colorize = True
        if resolution is not None:
            self.cfg.render_web.resolution = [resolution, resolution]
        latent_codes_fine_shape = feat_shape.to(self.device)
        latent_codes_fine_color = feat_color.to(self.device)
        if not hasattr(self, "renderer"):
            from edit3d.toolbox.colorsdf_renderer import SDFRenderer

            renderer = SDFRenderer(self.cfg.render_web, self.device, colorize)
        self.eval()
        with torch.no_grad():
            sdf_fun, _ = self._get_render_sdfs(latent_codes_fine_shape, latent_codes_fine_color)
            print("R", end="")
            img = renderer.render(sdf_fun, coloridx=None)
        return img

    # render the sketch
    def render_sketch(self, feature):
        feature = feature.to(self.device)
        self.eval()
        with torch.no_grad():
            img = self.imgen_net(feature)
            img = cv2.cvtColor(img[1].squeeze().cpu().numpy(), cv2.COLOR_GRAY2RGB)
        return img

    # render the color image
    def render_color2d(self, z_color, shape_feat):
        feature = torch.cat([z_color, shape_feat], dim=-1)
        feature = feature.to(self.device)
        self.eval()
        with torch.no_grad():
            img = self.colorgen_net(feature)
            img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        return img

    # used for editting, backprop gradient
    def forward_sketch(self, feature):
        feature = feature.to(self.device)
        with torch.no_grad():
            _, img = self.imgen_net(feature)
        return img

    # used for editting, backprop gradient
    def forward_color2d(self, z_color, shape_feat):
        feature = torch.cat([z_color, shape_feat], dim=-1)
        feature = feature.to(self.device)
        with torch.no_grad():
            img = self.colorgen_net(feature)
        return img

    def forward_color2d_grad(self, z_color, shape_feat):
        feature = torch.cat([z_color, shape_feat], dim=-1)
        feature = feature.to(self.device)
        img = self.colorgen_net(feature)
        return img

    def resume_demo(self, ckpt_path):
        print("WebDemo Resuming {}...".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.cfg.train_shape_ids = range(ckpt["trainer_state_dict"]["latent_embeddings_shape.weight_mu"].shape[0])
        self.prep_train()
        self.load_state_dict(ckpt["trainer_state_dict"], strict=False)
        self.sid2idx = ckpt["shapeid2idx"]
        # print(self.sid2idx)
