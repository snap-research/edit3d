import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder_shape(nn.Module):
    # shape decoder
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.dropout = cfg.dropout
        dropout_prob = cfg.dropout_prob
        self.use_tanh = cfg.use_tanh
        in_ch = cfg.in_ch
        out_ch = cfg.out_ch
        feat_ch = cfg.hidden_ch

        print(
            "[DeepSDF MLP-9] Dropout: {}; Do_prob: {}; in_ch: {}; hidden_ch: {}".format(
                self.dropout, dropout_prob, in_ch, feat_ch
            )
        )
        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_3 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_4 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Linear(feat_ch, out_ch)

        else:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
            )
            self.net2_1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_2 = nn.Sequential(
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_3 = nn.Sequential(
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.net2_4 = nn.Sequential(
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
            )
            self.out = nn.Sequential(
                nn.Dropout(dropout_prob, inplace=False), nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z, feat_layer=2):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)

        out2_1 = self.net2_1(in2)
        out2_2 = self.net2_2(out2_1)
        out2_3 = self.net2_3(out2_2)
        out2_4 = self.net2_4(out2_3)
        out2 = self.out(out2_4)

        if feat_layer == 1:
            feat = out2_1
        elif feat_layer == 2:
            feat = out2_2
        elif feat_layer == 3:
            feat = out2_3
        elif feat_layer == 4:
            feat = out2_4

        if self.use_tanh:
            out2 = torch.tanh(out2)

        return out2, feat


class Decoder(nn.Module):
    # shape + color decoder
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.use_tanh = cfg.use_tanh
        in_ch = cfg.color_in_ch + cfg.hidden_ch  # z_color + dim of feat
        feat_ch = cfg.hidden_ch
        self.fuse_layer = cfg.fuse_layer
        # mlp for shape
        self.shape_net = Decoder_shape(cfg)
        # mlp for color
        self.color_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 3),
        )

    def forward(self, z_shape, z_color):

        shape_out, shape_feat = self.shape_net(z_shape, self.fuse_layer)

        latent_color = torch.cat([z_color, shape_feat], dim=-1)
        color_out = self.color_net(latent_color)

        if self.use_tanh:
            color_out = torch.tanh(color_out)

        return shape_out, color_out
