import importlib
import os

# PyTorch
import torch
import torch.nn.functional as F

import edit3d.models.embeddings
import edit3d.toolbox.lr_scheduler
from edit3d.trainers.base_trainer import BaseTrainer
from edit3d.trainers.losses import laploss


def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD


class Trainer(BaseTrainer):
    def __init__(self, cfg, args, device):
        super(BaseTrainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.device = device

        # deepsdf model
        deepsdf_lib = importlib.import_module(cfg.models.deepsdf.type)
        self.deepsdf_net = deepsdf_lib.Decoder(cfg.models.deepsdf)
        if device == torch.device("cuda:0"):
            self.deepsdf_net.to(self.device)
        print("ShapeSDF Net:")
        print(self.deepsdf_net)
        # Init loss functions
        self.lossfun_fine = self._get_lossfun(self.cfg.trainer.loss_fine_shape)
        # Init optimizers
        self.optim_deepsdf, self.lrscheduler_deepsdf = self._get_optim(
            self.deepsdf_net.parameters(),  # model params
            self.cfg.trainer.optim_deepsdf,
        )  # hyper-params

        # colorsdf model
        colorsdf_lib = importlib.import_module(cfg.models.colorsdf.type)
        self.colorsdf_net = colorsdf_lib.Decoder(cfg.models.colorsdf)
        self.colorsdf_net.to(self.device)
        print("ColorSDF Net:")
        print(self.colorsdf_net)
        # Init loss functions
        self.lossfun_color3D = self._get_lossfun(self.cfg.trainer.loss_color3D)
        # Init optimizers
        self.optim_colorsdf, self.lrscheduler_colorsdf = self._get_optim(
            self.colorsdf_net.parameters(),  # model params
            self.cfg.trainer.optim_colorsdf,
        )  # hyper-params
        # sketch generator
        imgen_lib = importlib.import_module(cfg.models.im_gen.type)
        self.imgen_net = imgen_lib.Decoder(cfg.models.im_gen)
        self.imgen_net.to(self.device)
        print("Sketch Generator:")
        print(self.imgen_net)
        # Init loss functions
        self.lossfun_sketch = self._get_lossfun(self.cfg.trainer.loss_image)
        # Init optimizers
        self.optim_imgen, self.lrscheduler_imgen = self._get_optim(
            self.imgen_net.parameters(), self.cfg.trainer.optim_imgen  # model params
        )  # hyper-params

        # color image generator
        colorgen_lib = importlib.import_module(cfg.models.color_gen.type)
        self.colorgen_net = colorgen_lib.Decoder(cfg.models.color_gen)
        self.colorgen_net.to(self.device)
        print("Color Image Generator:")
        print(self.colorgen_net)
        # Init loss functions
        self.lossfun_color2D = self._get_lossfun(self.cfg.trainer.loss_color2D)
        # Init optimizers
        self.optim_colorgen, self.lrscheduler_colorgen = self._get_optim(
            self.colorgen_net.parameters(), self.cfg.trainer.optim_imgen  # model params
        )  # hyper-params
        self.additional_log_info = {}

    # Init training-specific contexts
    def _get_latent(self, cfg, N):
        embedding = getattr(edit3d.models.embeddings, cfg.type)
        embedding_instance = embedding(cfg, N=N, dim=cfg.dim).to(self.device)
        return embedding_instance

    def prep_train(self):  # czz: assign each instance a unique idx

        # self.cfg.train_shape_ids has content only in demo mode
        print(len(self.cfg.train_shape_ids))
        self.sid2idx = {k: v for v, k in enumerate(sorted(self.cfg.train_shape_ids))}
        print("[ImGen Trainer] init. #entries in sid2idx: {}".format(len(self.sid2idx)))

        # shape embedding
        self.latent_embeddings_shape = self._get_latent(self.cfg.trainer.latent_code_shape, N=len(self.sid2idx))
        print(self.latent_embeddings_shape)
        (   
            self.optim_latentcode_shape,
            self.lrscheduler_latentcode_shape,
        ) = self._get_optim(self.latent_embeddings_shape.parameters(), self.cfg.trainer.optim_latentcode)

        # color embedding
        self.latent_embeddings_color = self._get_latent(self.cfg.trainer.latent_code_color, N=len(self.sid2idx))

        (
            self.optim_latentcode_color,
            self.lrscheduler_latentcode_color,
        ) = self._get_optim(self.latent_embeddings_color.parameters(), self.cfg.trainer.optim_latentcode)

        self.train()

    # optimization
    def _get_optim(self, parameters, cfg):
        if cfg.type.lower() == "adam":
            optim = torch.optim.Adam(
                parameters,
                lr=cfg.lr,
                betas=cfg.betas,
                eps=cfg.eps,
                weight_decay=cfg.weight_decay,
                amsgrad=False,
            )
        elif cfg.type.lower() == "sgd":
            optim = torch.optim.SGD(
                parameters,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise NotImplementedError("Unknow optimizer: {}".format(cfg.type))
        scheduler = None
        if hasattr(cfg, "lr_scheduler"):
            scheduler = getattr(edit3d.toolbox.lr_scheduler, cfg.lr_scheduler.type)(cfg.lr_scheduler)
        return optim, scheduler

    # lr schedule
    def _step_lr(self, epoch):
        lr_latentcode_shape = self.lrscheduler_latentcode_shape(epoch)
        for g in self.optim_latentcode_shape.param_groups:
            g["lr"] = lr_latentcode_shape

        lr_latentcode_color = self.lrscheduler_latentcode_color(epoch)
        for g in self.optim_latentcode_color.param_groups:
            g["lr"] = lr_latentcode_color

        lr_deepsdf = self.lrscheduler_deepsdf(epoch)
        for g in self.optim_deepsdf.param_groups:
            g["lr"] = lr_deepsdf

        lr_colorsdf = self.lrscheduler_colorsdf(epoch)
        for g in self.optim_colorsdf.param_groups:
            g["lr"] = lr_colorsdf

        lr_imgen = self.lrscheduler_imgen(epoch)
        for g in self.optim_imgen.param_groups:
            g["lr"] = lr_imgen

        lr_colorgen = self.lrscheduler_colorgen(epoch)
        for g in self.optim_colorgen.param_groups:
            g["lr"] = lr_colorgen

    # loss function
    def _get_lossfun(self, cfg):
        print(cfg)
        if cfg.type.lower() == "clamped_l1":
            from edit3d.models.lossfuns import clamped_l1

            lossfun = lambda pred, gt: torch.mean(clamped_l1(pred, gt, trunc=cfg.trunc), dim=-1)
        elif cfg.type.lower() == "clamped_l1_correct":
            from edit3d.models.lossfuns import clamped_l1_correct as clamped_l1

            lossfun = lambda pred, gt: clamped_l1(pred, gt, trunc=cfg.trunc)
        elif cfg.type.lower() == "l1":
            lossfun = lambda pred, gt: torch.mean(torch.abs(pred - gt), dim=-1)
        elif cfg.type.lower() == "onesided_l2":
            from edit3d.models.lossfuns import onesided_l2

            lossfun = onesided_l2
        elif cfg.type.lower() == "mse":
            from edit3d.models.lossfuns import mse

            lossfun = mse
        elif cfg.type.lower() == "binary_cross_entropy":
            from edit3d.models.lossfuns import binary_cross_entropy

            lossfun = binary_cross_entropy
        else:
            raise NotImplementedError("Unknow loss function: {}".format(cfg.type))
        return lossfun

    # Convert list of shape ids to their corresponding indices in embedding.
    def _b_sid2idx(self, sid_list):
        data_indices = torch.tensor([self.sid2idx[x] for x in sid_list], dtype=torch.long, device=self.device)
        return data_indices

    def _b_idx2latent(self, latent_embeddings, indices, num_augment_pts=None):

        #forward pass VADLogVar Embedding
        batch_latent_dict = latent_embeddings(indices, num_augment_pts=num_augment_pts)
        
        batch_latent = batch_latent_dict["latent_code"]
        if "mu" in batch_latent_dict.keys() and "logvar" in batch_latent_dict.keys():
            batch_mu = batch_latent_dict["mu"]
            batch_logvar = batch_latent_dict["logvar"]
            kld = KLD(batch_mu, batch_logvar)
            self.additional_log_info["vad_batch_mu_std"] = torch.std(batch_mu).item()
            self.additional_log_info["vad_batch_kld"] = kld.item()
            if "std" in batch_latent_dict.keys():
                batch_sigma = batch_latent_dict["std"]
            else:
                batch_sigma = torch.exp(0.5 * batch_logvar)
            self.additional_log_info["vad_batch_sigma_mean"] = torch.mean(batch_sigma).item()
        else:
            kld = 0.0
        if "latent_code_augment" in batch_latent_dict.keys():
            batch_latent_aug = batch_latent_dict["latent_code_augment"]
        else:
            batch_latent_aug = batch_latent
        return (
            batch_latent,
            batch_latent_aug,
            kld,
        )  # by default, batch_latent is simply the mu

    """ ---------------------------  forward functions ----------------------- """

    def _forward_deepsdf(self, z, p):
        bs = z.size(0)
        N = p.size(1)
        if len(z.shape) == 2:
            z = z.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([z, p], dim=-1)
        dists = self.deepsdf_net(inp)  # [64 2048 1]
        return dists

    def _forward_imgen(self, z):
        return self.imgen_net(z)

    def _forward_colorgen(self, z_color, shape_feat):
        latent = torch.cat([z_color, shape_feat], dim=-1)
        return self.colorgen_net(latent)

    def _forward_colorsdf(self, z_color, shape_feat, p):

        latent = torch.cat([z_color, shape_feat], dim=-1)
        bs = latent.size(0)
        N = p.size(1)
        if len(latent.shape) == 2:
            latent = latent.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([latent, p], dim=-1)
        dists = self.colorsdf_net(inp)  # [64 2048 3] # rgb
        return dists

    """ ---------------------------  training procedure ----------------------- """

    def epoch_start(self, epoch):
        # Setting LR
        self.train()
        self._step_lr(epoch)
        self.optim_latentcode_shape.zero_grad()
        self.optim_latentcode_color.zero_grad()

    def step(self, data):

        data_ids = data["shape_ids"]
        data_f = data["surface_samples"].to(self.device, non_blocking=True)  # [64 2048 7] xyzd+rgb
        data_indices = data["shape_indices"].squeeze(-1).to(self.device, non_blocking=True)  # [64]
        data_sketch = data["sketch"].to(self.device, non_blocking=True)
        data_color2d = data["color_2d"].to(self.device, non_blocking=True)
        # data_color3d = data['color_3d'].to(self.device, non_blocking=True).to(torch.float)

        # shape and color have different latent code
        (latent_codes_coarse_shape, latent_codes_fine_shape, kld_shape,) = self._b_idx2latent(
            self.latent_embeddings_shape, data_indices, num_augment_pts=data_f.size(1)
        )  # [64 128]

        (latent_codes_coarse_color, latent_codes_fine_color, kld_color,) = self._b_idx2latent(
            self.latent_embeddings_color, data_indices, num_augment_pts=data_f.size(1)
        )  # [64 128]

        # for sketch, use latent_code_coarse whose shape is (64, 128)
        # for handles, use latent_code_coarse whose shape is (64, 128)
        # for deepsdf, use latent_code_fine whose shape is (64, 2048,128)
        # we just need to reserve one of {latent_code_fine, latent_code_coarse} to backprop to mu and std

        # latent_codes_fine = latent_codes_fine.detach()
        latent_codes_coarse_shape = latent_codes_coarse_shape.detach()
        latent_codes_coarse_color = latent_codes_coarse_color.detach()

        # because the output of the colorsdf is positive, assume undefined region has white color (1,1,1)
        data_f[..., 4:7][data_f[..., 4:7] < 0] *= -1

        # DeepSDF
        self.optim_deepsdf.zero_grad()
        pts_fine = data_f[..., :3]
        dists_gt_fine = data_f[..., [3]].squeeze(-1)
        dists_deepsdf, shape_feat = self._forward_deepsdf(latent_codes_fine_shape, pts_fine)  # 64, 2048, 1
        dists_deepsdf = dists_deepsdf.squeeze(-1)
        loss_fine_shape = torch.mean(self.lossfun_fine(dists_deepsdf, dists_gt_fine))

        # ColorSDF
        self.optim_colorsdf.zero_grad()
        color_gt_fine = data_f[..., 4:7].squeeze(-1)
        if self.cfg.trainer.color_shape_joint:
            rgb_colorsdf = self._forward_colorsdf(latent_codes_fine_color, shape_feat, pts_fine).squeeze(-1)
        else:
            rgb_colorsdf = self._forward_colorsdf(latent_codes_fine_color, shape_feat.detach(), pts_fine).squeeze(-1)
        loss_color3D = torch.mean(self.lossfun_color3D(color_gt_fine, rgb_colorsdf))

        # sketch generator
        self.optim_imgen.zero_grad()
        im_logits, im_samples = self._forward_imgen(latent_codes_coarse_shape)
        loss_sketch = torch.mean(self.lossfun_sketch(im_logits, data_sketch))

        # color image generator
        self.optim_colorgen.zero_grad()
        im_samples = self._forward_colorgen(latent_codes_coarse_color, latent_codes_coarse_shape)
        # loss_color2D = torch.mean(self.lossfun_color2D(im_samples, data_color2d))
        lap_loss = laploss(im_samples, data_color2d)
        mse_loss = F.mse_loss(im_samples, data_color2d)
        loss_color2D = lap_loss + mse_loss

        loss = 0.5 * (
            loss_fine_shape * self.cfg.trainer.loss_fine_shape.weight
            + loss_sketch * self.cfg.trainer.loss_image.weight
            + loss_color3D * self.cfg.trainer.loss_color3D.weight
            + loss_color2D * self.cfg.trainer.loss_color2D.weight
        )

        (
            loss + kld_shape * self.cfg.trainer.kld_weight_shape + kld_color * self.cfg.trainer.kld_weight_color
        ).backward()

        self.optim_deepsdf.step()
        self.optim_colorsdf.step()
        self.optim_imgen.step()
        self.optim_colorgen.step()

        loss_fine_shape = loss_fine_shape.detach().item()
        loss_color3D = loss_color3D.detach().item()
        loss_sketch = loss_sketch.detach().item()
        loss_color2D = loss_color2D.detach().item()

        log_info = {
            "loss": loss.item(),
            "loss_sketch": loss_sketch,
            "loss_shape": loss_fine_shape,
            "loss_color3D": loss_color3D,
            "loss_color2D": loss_color2D,
        }

        log_info.update(self.additional_log_info)
        return log_info

    def epoch_end(self, epoch, **kwargs):
        self.optim_latentcode_shape.step()
        self.optim_latentcode_color.step()

    def sample_images(self, data):

        data_indices = data["shape_indices"].squeeze(-1).to(self.device, non_blocking=True)  # [64]

        # sample sketch
        data_sketch = data["sketch"].squeeze(-1).to(self.device, non_blocking=True)
        (latent_codes_coarse_shape, latent_codes_fine_shape, kld_shape,) = self._b_idx2latent(
            self.latent_embeddings_shape, data_indices, num_augment_pts=2048
        )  # [64 128]

        latent_codes_coarse_shape = latent_codes_coarse_shape.detach()
        with torch.no_grad():
            _, sketch_samples = self._forward_imgen(latent_codes_coarse_shape)

        # sample color images
        data_color = data["color_2d"].squeeze(-1).to(self.device, non_blocking=True)
        latent_codes_coarse_color, latent_codes_fine_color, kld = self._b_idx2latent(
            self.latent_embeddings_color, data_indices, num_augment_pts=2048
        )  # [64 128]

        latent_codes_coarse_color = latent_codes_coarse_color.detach()
        with torch.no_grad():
            rgb_samples = self._forward_colorgen(latent_codes_coarse_color, latent_codes_coarse_shape)

        # sample sdf with color info
        rendered_imgs = []
        for idx in range(latent_codes_coarse_shape.shape[0]):
            rendered_img = self.render_express(latent_codes_coarse_shape[idx], latent_codes_coarse_color[idx])
            rendered_imgs.append(rendered_img)

        return {
            "gt_sketch": data_sketch,
            "gen_sketch": sketch_samples,
            "gt_color": data_color,
            "gen_color": rgb_samples,
            "render_sdf": rendered_imgs,
        }

    def _get_render_sdfs(self, zz_shape, zz_color):
        def sdf_fun(p):  # p: N, 3
            N = p.size(0)
            p = p.unsqueeze(0)

            z_shape = zz_shape
            if len(z_shape.shape) == 1:
                z_shape = z_shape.unsqueeze(0)
            if len(z_shape.shape) == 2:
                z_shape = z_shape.unsqueeze(1)
            z_shape = z_shape.expand(-1, N, -1)
            inp = torch.cat([z_shape, p], dim=-1)

            dists, shape_feats = self.deepsdf_net(inp)  # [1 N 1]
            dists = dists.reshape(-1, 1)

            z_color = zz_color
            if len(z_color.shape) == 1:
                z_color = z_color.unsqueeze(0)
            if len(z_color.shape) == 2:
                z_color = z_color.unsqueeze(1)
            z_color = z_color.expand(-1, N, -1)
            inp = torch.cat([z_color, shape_feats, p], dim=-1)
            with torch.no_grad():
                color3d = self.colorsdf_net(inp)  # [1 N 1]
            color3d = color3d.reshape(-1, 3)
            return {"dists": dists, "color3d": color3d}

        return sdf_fun

    def render_express(self, feat_shape, feat_color):

        latent_codes_fine_shape = feat_shape.to(self.device)
        latent_codes_fine_color = feat_color.to(self.device)
        if not hasattr(self, "renderer"):
            from edit3d.toolbox.colorsdf_renderer import SDFRenderer

            renderer = SDFRenderer(self.cfg.render_web, self.device)
        self.eval()
        with torch.no_grad():
            sdf_fun = self._get_render_sdfs(latent_codes_fine_shape, latent_codes_fine_color)
            print("R", end="")
            img = renderer.render(sdf_fun, coloridx=None)
            # img = img[...,[2,1,0]] # RGB -> BGR
        self.train()
        return img

    # save checkpoints
    def save(self, epoch, step):
        save_name = "epoch_{}_iters_{}.pth".format(epoch, step)
        path = os.path.join(self.cfg.save_dir, save_name)
        torch.save(
            {
                "trainer_state_dict": self.state_dict(),
                "optim_latentcode_shape_state_dict": self.optim_latentcode_shape.state_dict(),  # latent for shape
                "optim_latentcode_color_state_dict": self.optim_latentcode_color.state_dict(),  # latent for color
                "optim_imgen_state_dict": self.optim_imgen.state_dict(),  # optim for sketch
                "optim_deepsdf_state_dict": self.optim_deepsdf.state_dict(),  # optim for 3D shape
                "optim_colorgen_state_dict": self.optim_colorgen.state_dict(),  # optim for 2D color
                "optim_colorsdf_state_dict": self.optim_colorsdf.state_dict(),  # optim for 3D color
                "shapeid2idx": self.sid2idx,  # shape id
                "epoch": epoch,
                "step": step,
            },
            path,
        )

    # resume from training
    def resume(self, ckpt_path):
        print("Resuming {}...".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt["trainer_state_dict"], strict=False)
        return ckpt["epoch"]
