import os
import numpy as np
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import deep_sdf
import argparse
import yaml
import importlib
import cv2

def save(trainer, latent, target, mask, outdir, imname):
    mesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + '.pth')
    pred_sketch_filename = os.path.join(outdir, imname + '_sketch.png')
    pred_3D_filename = os.path.join(outdir, imname + '_3D.png')
    target_filename = os.path.join(outdir, imname + '_target.png')
    masked_target_filename = os.path.join(outdir, imname + '_masked_target.png')
    #with torch.no_grad():
    #    deep_sdf.mesh.create_mesh(trainer.deepsdf_net, latent, mesh_filename, 
    #                          N=256, max_batch=int(2 ** 18))
    torch.save(latent, latent_filename) 
    pred_sketch = trainer.render_sketch(latent)
    pred_3D     = trainer.render_express(latent)
    save_image(pred_sketch, pred_sketch_filename)
    cv2.imwrite(pred_3D_filename, cv2.cvtColor(pred_3D, cv2.COLOR_RGB2BGR))
    save_image(target.squeeze().cpu().numpy(), target_filename)
    if mask is not None:
        save_image((mask * target).squeeze().cpu().numpy(), masked_target_filename)


def is_exist(outdir, imname):
    mesh_filename = os.path.join(outdir, imname + '.ply')
    latent_filename = os.path.join(outdir, imname + '.pth')
    pred_sketch_filename = os.path.join(outdir, imname + '_sketch.png')
    pred_3D_filename = os.path.join(outdir, imname + '_3D.png')
    target_filename = os.path.join(outdir, imname + '_target.png')
    if (os.path.exists(mesh_filename) and os.path.exists(latent_filename) and os.path.exists(pred_sketch_filename) and os.path.exists(pred_3D_filename) and os.path.exists(target_filename)):
        return True
    else:
        return False

def save_image(image, outname):
    out = np.uint8(image * 255)
    cv2.imwrite(outname, out)

def load_image(path, imsize=128):
    transform = transforms.Compose([transforms.Resize((imsize,imsize)), transforms.ToTensor(),])
    data_im = Image.open(path)
    data = transform(data_im)
    return data

def reconstruct(trainer, target, mask, epoch, trial, gamma, beta, K=5):
    template, _ = trainer.get_known_latent(0) 
    latents = []
    losses = []
    for i in range(trial): # multi-trial latent optimization 
        init_latent = torch.randn_like(template).to(template.device)
        latent, loss = trainer.step_manip_sketch(init_latent, target, mask=mask, 
                                epoch=epoch, gamma=gamma, beta=beta)
        latents.append(latent[-1])
        losses.append(loss)
    # get the top K latent
    losses = np.array(losses)
    indices = losses.argsort()[-K:]
    best_latents = [latents[idx] for idx in indices]
    return best_latents

def get_args():
    parser = argparse.ArgumentParser(description='Reconstruction')
    parser.add_argument('config', type=str, help='The configuration file.')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model checkpoint')
    parser.add_argument('--outdir', default=None, type=str, help='path of output')
    parser.add_argument('--impath', default=None, type=str, help='path of an image')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--mask-level', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.02, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--epoch', default=501, type=int)
    parser.add_argument('--trial', default=20, type=int)
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = dict2namespace(config)
    return args, config


def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')    
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)
    trainer.resume_demo(args.pretrained)
    idx2sid = {}
    for k, v in trainer.sid2idx.items():
        idx2sid[v] = k
    trainer.eval()
    def getname(impath):
        return impath.split("/")[-2]
    os.makedirs(args.outdir, exist_ok=True)
    imname = getname(impath)
    print("Reconstruct 3D from %s ..." % imname)
    target = load_image(impath)
    if args.mask: # reconstruct from partial views 
        mask = torch.zeros_like(target)
        C,H,W = mask.shape
        non_bg = torch.sum(target[0], dim=1) < W
        first,last = head_tail(non_bg)
        length = int((last - first) * args.mask_level)
        mask[:, first:first+length, :] = 1
    else: # reconstruct from full views
        mask = None
    latents = reconstruct(trainer, target, mask, args.epoch, args.trial, args.gamma, args.beta)
    for idx, latent in enumerate(latents):
        try:
            save(trainer, latent, target, mask, args.outdir, imname+f"_{idx}")
        except Exception:
            pass

if __name__ == "__main__":
    args, cfg = get_args()
    main(args, cfg)

