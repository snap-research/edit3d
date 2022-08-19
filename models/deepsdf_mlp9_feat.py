import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    # shape decoder 
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.dropout = cfg.dropout
        dropout_prob = cfg.dropout_prob
        self.use_tanh = cfg.use_tanh
        in_ch = cfg.in_ch
        out_ch = cfg.out_ch
        feat_ch = cfg.hidden_ch
        self.feat_layer = cfg.feat_layer # out the act of layer x as feat of shape 

        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True)
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
            self.out =  nn.Sequential(
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print('[num parameters: {}]'.format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        feat_layer = self.feat_layer

        out2_1 = self.net2_1(in2)
        out2_2 = self.net2_2(out2_1)
        out2_3 = self.net2_3(out2_2)
        out2_4 = self.net2_4(out2_3)
        out2    = self.out(out2_4)

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
 
