#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]

    if latent_vector is None:
        inputs = queries
    else:
        latent_repeat = latent_vector.expand(num_samples, -1)
        inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)

    return sdf

# based on the customized colorsdf network
def decode_colorsdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)
    inputs = torch.cat([latent_repeat, queries], 1)
    sdf, _ = decoder(inputs)
    return sdf


# based on the customized colorsdf network
def decode_colorsdf2(deepsdf, colorsdf, shape_code, color_code, queries):
    num_samples = queries.shape[0]
    shape_codes = shape_code.expand(num_samples, -1)
    color_codes = color_code.expand(num_samples, -1)
    inputs = torch.cat([shape_codes, queries], 1)
    sdf, shape_feats = deepsdf(inputs)
    inputs2 = torch.cat([color_codes, shape_feats, queries], dim=-1)
    color3d = colorsdf(inputs2)
    color3d = color3d[:, [2,1,0]]
    return sdf, color3d

