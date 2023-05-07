import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class VADLogVar(nn.Module):
    def __init__(self, cfg, N, dim):
        super(VADLogVar, self).__init__()
        self.cfg = cfg
        self.N = N
        self.dim = dim
        self.weight_mu = nn.Parameter(torch.Tensor(N, dim))
        self.weight_logvar = nn.Parameter(torch.Tensor(N, dim))
        self.reset_parameters()
        print("[VADLogVar Embedding] #entries: {}; #dims: {}; cfg: {}".format(self.N, self.dim, self.cfg))

    def reset_parameters(self):
        if self.cfg.mu_init_std is None:
            mu_init_std = 1.0 / np.sqrt(self.dim)
        else:
            mu_init_std = self.cfg.mu_init_std
        torch.nn.init.normal_(
            self.weight_mu.data,
            0.0,
            mu_init_std,
        )
        if self.cfg.logvar_init_std is None:
            logvar_init_std = 1.0 / np.sqrt(self.dim)
        else:
            logvar_init_std = self.cfg.logvar_init_std
        torch.nn.init.normal_(
            self.weight_logvar.data,
            self.cfg.logvar_init_mean,
            logvar_init_std,
        )

    def forward(self, idx, **kwargs):
        num_augment_pts = kwargs["num_augment_pts"]
        mu = self.weight_mu[idx]
        logvar = self.weight_logvar[idx]
        if self.cfg.fix_var:
            logvar = logvar.detach()
        std = torch.exp(0.5 * logvar)
        if self.training:
            # print('[VADLogVar Embedding] Train mode forward')
            eps = torch.randn_like(std)
            batch_latent = mu + eps * std
            if self.cfg.augment_latent:
                eps_aug = torch.randn(std.size(0), num_augment_pts, std.size(1), device=std.device)
                batch_latent_aug = mu.unsqueeze(1) + eps_aug * std.unsqueeze(
                    1
                )  # shape = (batch_sz, num_pts, latent_dim)
            elif self.cfg.sample_twice:
                eps_aug = torch.randn_like(std)
                batch_latent_aug = mu + eps_aug * std
            else:
                batch_latent_aug = batch_latent
            return {
                "latent_code": batch_latent,
                "latent_code_augment": batch_latent_aug,
                "mu": mu,
                "logvar": logvar,
                "std": std,
            }
        else:
            print("[VADLogVar Embedding] Test mode forward")
            batch_latent = mu
            return {"latent_code": batch_latent, "mu": mu, "logvar": logvar, "std": std}

    # randomly sample from the latent space and calculate the FID or check the quality of samples
    def random_sample(self, num_pts):
        mu = torch.mean(self.weight_mu, dim=0)
        std = torch.std(self.weight_mu, dim=0)
        eps = torch.randn(num_pts, std.size(0), device=std.device)
        batch_latent = mu.unsqueeze(0) + eps * std.unsqueeze(0)
        return {"latent_code": batch_latent, "mu": mu, "std": std}

    # randomly sample from the latent space and calculate the FID or check the quality of samples
    def random_sample_near_mu(self, num_pts=20, num_aug=10):
        mus = []
        batch_latents = []
        for idx in range(num_pts):
            mu = self.weight_mu[idx]
            logvar = self.weight_logvar[idx]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(num_aug, std.size(0), device=std.device)
            batch_latent = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # N * 128
            mus.append(mu)
            batch_latents.append(batch_latent)
        return {"latent_code": batch_latents, "mu": mus}

    def random_sample_gaussian(self, num_pts):
        mu = self.weight_mu[0]
        logvar = self.weight_logvar[0]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_pts, std.size(0), device=std.device)
        return {"latent_code": eps}


class AD(nn.Module):
    def __init__(self, cfg, N, dim):
        super(AD, self).__init__()
        self.cfg = cfg
        self.N = N
        self.dim = dim
        self.embed_params = nn.Parameter(torch.Tensor(N, dim))
        self.reset_parameters()
        print("[AD Embedding] #entries: {}; #dims: {}; cfg: {}".format(self.N, self.dim, self.cfg))

    def reset_parameters(self):
        if self.cfg.init_std is None:
            init_std = 1.0 / np.sqrt(self.dim)
        else:
            init_std = self.cfg.init_std
        torch.nn.init.normal_(
            self.embed_params.data,
            0.0,
            init_std,
        )

    def _normalize(self, idx):
        batch_embed = self.embed_params[idx].detach()
        batch_norms = torch.sqrt(torch.sum(batch_embed.data ** 2, dim=-1, keepdim=True))
        batch_scale_factors = torch.clamp(batch_norms, self.cfg.max_norm, None)
        batch_embed_normalized = batch_embed / batch_scale_factors
        self.embed_params.data[idx] = batch_embed_normalized

    def forward(self, idx, **kwargs):
        if getattr(self.cfg, "max_norm", None):
            self._normalize(idx)
        batch_embed = self.embed_params[idx]

        return {"latent_code": batch_embed}
