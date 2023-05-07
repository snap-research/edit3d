import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import deep_sdf
import argparse
import yaml
import importlib
import cv2
import logging

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=logging.getLevelName(log_level))

logger = logging.getLogger(__name__)
CUDA_DEVICE = "cuda:0"

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() and os.getenv("USE_GPU") else "cpu")


def save(trainer, latents, target, mask, outdir, imname):
    colormesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + ".pth")
    pred_rgb_filename = os.path.join(outdir, imname + "_rgb.png")
    pred_sketch_filename = os.path.join(outdir, imname + "_sketch.png")
    pred_3D_filename = os.path.join(outdir, imname + "_3D.png")
    target_filename = os.path.join(outdir, imname + "_target.png")
    masked_target_filename = os.path.join(outdir, imname + "_masked_target.png")
    shape_code, color_code = latents
    with torch.no_grad():
        deep_sdf.colormesh.create_mesh(
            trainer.deepsdf_net,
            trainer.colorsdf_net,
            shape_code,
            color_code,
            colormesh_filename,
            N=256,
            max_batch=int(2 ** 18),
            device=device,
        )

    torch.save(latents, latent_filename)
    pred_rgb = trainer.render_color2d(color_code, shape_code)
    save_image(pred_rgb, pred_rgb_filename)

    pred_sketch = trainer.render_sketch(shape_code)
    save_sketch(pred_sketch, pred_sketch_filename)

    pred_3D = trainer.render_express(shape_code, color_code, resolution=512)
    pred_3D = cv2.cvtColor(pred_3D, cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_3D_filename, pred_3D)

    save_image(np.moveaxis(target.squeeze().cpu().numpy(), 0, -1), target_filename)
    if mask is not None:
        save_image(
            np.moveaxis((mask * target).squeeze().cpu().numpy(), 0, -1),
            masked_target_filename,
        )


def exists(outdir, imname):
    mesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + ".pth")
    pred_rgb_filename = os.path.join(outdir, imname + "_rgb.png")
    pred_3D_filename = os.path.join(outdir, imname + "_3D.png")
    target_filename = os.path.join(outdir, imname + "_target.png")
    if (
        os.path.exists(mesh_filename)
        and os.path.exists(latent_filename)
        and os.path.exists(pred_rgb_filename)
        and os.path.exists(pred_3D_filename)
        and os.path.exists(target_filename)
    ):
        return True
    else:
        return False


def save_image(image, outname):
    out = np.uint8(image * 255)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outname, out)


def save_sketch(image, outname):
    out = np.uint8(image * 255)
    cv2.imwrite(outname, out)


def load_image(path, imsize=128):
    transform = transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
        ]
    )
    data_im = Image.open(path)
    x = np.array(data_im)
    if x.shape[-1] == 4:
        color = (255, 255, 255)
        r, g, b, a = np.rollaxis(x, axis=-1)
        r[a == 0] = color[0]
        g[a == 0] = color[1]
        b[a == 0] = color[2]
        x = np.dstack([r, g, b])
        data_im = Image.fromarray(x, "RGB")
    data = transform(data_im)
    return data


def load_image_photoshop(path, imsize=128):
    """Load images preprocessed by photoshop (e.g. center cropping, resizing etc.)"""
    transform = transforms.Compose(
        [
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
        ]
    )
    im = cv2.imread(path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    data_im = Image.fromarray(im_rgb, "RGB")
    data = transform(data_im)
    return data


def reconstruct(trainer, target, mask, epoch, trial, gamma, beta, device):
    temp_shape, temp_color = trainer.get_known_latent(0)
    min_loss = np.inf
    for i in range(trial):  # multi-trial latent optimization
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


def head_tail(data):
    head = 0
    tail = len(data) - 1
    for i, item in enumerate(data):
        if item:
            head = i
            break
    for i, item in enumerate(data):
        if item:
            tail = i
    return head, tail


def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)
    trainer.resume_demo(args.pretrained)
    idx2sid = {}
    for k, v in trainer.sid2idx.items():
        idx2sid[v] = k
    trainer.eval()
    os.makedirs(args.outdir, exist_ok=True)

    impath = args.impath
    imname = os.path.split(impath)[1].split(".")[0]
    logger.info("Reconstruct 3D from %s ..." % imname)
    target = load_image(impath)

    if args.mask:  # reconstruct from partial views
        mask = torch.zeros_like(target)
        C, H, W = mask.shape
        non_bg = torch.sum(target[0], dim=1) < W
        first, last = head_tail(non_bg)
        length = int((last - first) * args.mask_level)
        mask[:, first : first + length, :] = 1
    else:  # reconstruct from full views
        mask = None
    latents = reconstruct(trainer, target, mask, args.epoch, args.trial, args.gamma, args.beta, device)
    try:
        save(trainer, latents, target, mask, args.outdir, imname)
    except Exception:
        logger.error("Could not save output.", exc_info=True)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction")
    parser.add_argument("config", type=str, help="The configuration file.")
    parser.add_argument("--pretrained", default=None, type=str, help="pretrained model checkpoint")
    parser.add_argument("--outdir", default=None, type=str, help="path of output")
    parser.add_argument("--impath", default=None, type=str, help="path of an image")
    parser.add_argument("--mask", default=False, action="store_true")
    parser.add_argument("--mask-level", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.02, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--epoch", default=1001, type=int)
    parser.add_argument("--trial", default=20, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    main(args, config)
