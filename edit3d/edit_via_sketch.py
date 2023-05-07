import argparse
import glob
import importlib
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from edit3d import device, logger
from edit3d.models import deep_sdf
from edit3d.utils.utils import dict2namespace


def save(trainer, latent, target, outdir, imname, save_ply=False):
    """Save 2D and 3D modalities after editing"""
    colormesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + ".pth")
    pred_sketch_filename = os.path.join(outdir, imname + "_sketch.png")
    pred_3D_filename = os.path.join(outdir, imname + "_3D.png")
    target_filename = os.path.join(outdir, imname + "_target.png")
    shape_code, color_code = latent
    torch.save(latent, latent_filename)
    if save_ply:
        with torch.no_grad():
            deep_sdf.colormesh.create_mesh(
                trainer.deepsdf_net,
                trainer.colorsdf_net,
                shape_code,
                color_code,
                colormesh_filename,
                N=256,
                max_batch=int(2 ** 18),
            )
    pred_3d = trainer.render_express(shape_code, color_code, resolution=256)
    pred_3d = cv2.cvtColor(pred_3d, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_3D_filename, pred_3d)
    pred_sketch = trainer.render_sketch(shape_code)
    save_image(pred_sketch, pred_sketch_filename)
    save_image(target.squeeze().cpu().numpy(), target_filename)


def save_init(trainer, latent, outdir, imname, colormesh=True):
    """Save 2D and 3D modalities before editing"""
    colormesh_filename = os.path.join(outdir, imname)
    mesh_filename = os.path.join(outdir, imname + "_wocolor")
    latent_filename = os.path.join(outdir, imname + ".pth")
    pred_3D_filename = os.path.join(outdir, imname + "_3D.png")
    pred_wocolor_3D_filename = os.path.join(outdir, imname + "_wocolor_3D.png")
    shape_code, color_code = latent
    if colormesh:  # generate mesh with surface color from 3D colornet
        with torch.no_grad():
            deep_sdf.colormesh.create_mesh(
                trainer.deepsdf_net,
                trainer.colorsdf_net,
                shape_code,
                color_code,
                colormesh_filename,
                N=256,
                max_batch=int(2 ** 18),
            )
    else:  # generate mesh with default color
        with torch.no_grad():
            deep_sdf.mesh.create_mesh(
                trainer.deepsdf_net,
                shape_code,
                mesh_filename,
                N=256,
                max_batch=int(2 ** 18),
            )
    torch.save(latent, latent_filename)
    pred_3d_nocolor = trainer.render_express(shape_code, resolution=512)
    pred_3d_nocolor = cv2.cvtColor(pred_3d_nocolor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_wocolor_3D_filename, pred_3d_nocolor)
    pred_3d = trainer.render_express(shape_code, color_code, resolution=512)
    pred_3d = cv2.cvtColor(pred_3d, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_3D_filename, pred_3d)
    pred_sketch = trainer.render_sketch(shape_code)
    pred_sketch_filename = os.path.join(outdir, imname + "_sketch.png")
    save_image(pred_sketch, pred_sketch_filename)


def save_image(image, outname):
    out = np.uint8(image * 255)
    cv2.imwrite(outname, out)


def reconstruct(trainer, target, mask, epoch, trial, gamma, beta):
    temp_shape, temp_color = trainer.get_known_latent(0)
    min_loss = np.inf
    best_latent = None
    for i in range(trial):
        init_shape = torch.randn_like(temp_shape).to(device)
        init_color = torch.randn_like(temp_color).to(device)
        latent, loss = trainer.step_recon_rgb(
            init_shape,
            init_color,
            target,
            mask=mask,
            epoch=epoch,
            gamma=gamma,
            beta=beta,
        )
        if min_loss > loss:
            best_latent = latent[-1]
            min_loss = loss
    return best_latent


def get_mask(source, target):
    mask = 1.0 * ((target - source) != 0)  # 0 means the unmodified region
    return mask


def edit(trainer, init_latent, source, target, epoch, gamma, beta):
    since = time.time()
    init_shape, init_color = init_latent
    mask = get_mask(source, target)
    latent, loss = trainer.step_edit_sketch(init_shape, target, mask=mask, epoch=epoch, gamma=gamma, beta=beta)
    logger.info(f"Editing shape takes {time.time() - since} seconds")
    return latent, init_color  # here the latent contains multiple snapshot


def load_image(path, imsize=128):
    transform = transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
        ]
    )
    im = cv2.resize(cv2.imread(path), (128, 128))[:, :, 0]
    data_im = Image.fromarray(im)
    data = transform(data_im)
    return data


def load_image_and_sketch(source_path, editid, prefix):
    imagelist = glob.glob(os.path.join(source_path, f"{prefix}*_{editid}.png"))
    if len(imagelist) == 0:
        return None
    source_image = os.path.join(source_path, prefix + ".png")
    source_im = load_image(source_image)
    target_image = imagelist[0]
    target_im = load_image(target_image)
    return {"source": source_im, "target": target_im}


def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)
    trainer.resume_demo(args.pretrained)
    idx2sid = {}
    for k, v in trainer.sid2idx.items():
        idx2sid[v] = k
    trainer.eval()
    source_dir = os.path.abspath(args.source_dir)

    os.makedirs(args.outdir, exist_ok=True)

    if args.category == "airplane":
        prefix = "sketch-T-2"
    elif args.category == "chair":
        prefix = "sketch-F-2"
    else:
        logger.error("Only airplane and chair are supported categories")
        raise Exception(f"No such category: {args.category}")

    for imname in os.listdir(source_dir):

        source_path = os.path.join(source_dir, imname)
        logger.info("Edit 3D from %s ..." % source_path)
        for editid in range(1, 10):
            logger.debug(editid)
            data = load_image_and_sketch(source_dir, editid, prefix)
            if data is None or (imname not in trainer.sid2idx.keys()):
                continue

            targetdir = os.path.join(args.outdir, imname, str(editid))
            os.makedirs(targetdir, exist_ok=True)

            # save init
            source_latent = trainer.get_known_latent(trainer.sid2idx[imname])
            initdir = os.path.join(targetdir, "init")
            os.makedirs(initdir, exist_ok=True)
            save_init(trainer, source_latent, initdir, imname + "_init")

            # editing
            edit_latent, color_code = edit(
                trainer,
                source_latent,
                data["source"],
                data["target"],
                args.epoch,
                args.gamma,
                args.beta,
            )
            for iteration, latent_snap in enumerate(edit_latent[:10]):
                save(
                    trainer,
                    (latent_snap, color_code),
                    data["target"],
                    targetdir,
                    imname + f"_{iteration}",
                    save_ply=False,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction")
    parser.add_argument("config", type=str, help="The configuration file.")
    parser.add_argument("--pretrained", default=None, type=str, help="pretrained model checkpoint")
    parser.add_argument("--outdir", default=None, type=str, help="path of output")
    parser.add_argument("--category", default="airplane", type=str, help="path of output")
    parser.add_argument("--source_dir", default=None, type=str, help="a text file the lists image")
    parser.add_argument("--trial", default=20, type=int)
    parser.add_argument("--editid", default=1, type=int)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.02, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict2namespace(cfg)

    main(args, config)
