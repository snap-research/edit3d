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
import glob
import time


# arguments: color -- the random sampled color for diferent scribbles
def save(trainer, latent, source, target, scribble, mask,color,outdir, imname, save_ply=True):
    colormesh_filename = os.path.join(outdir, imname)
    # latent_filename = os.path.join(outdir, imname + '.pth')
    # pred_rgb_filename = os.path.join(outdir, imname + '_rgb.png')
    pred_3D_filename = os.path.join(outdir, imname + '_3D.png')
    # source_filename = os.path.join(outdir, imname + '_source.png')
    target_filename = os.path.join(outdir, imname + '_target.png')
    scribble_filename = os.path.join(outdir, imname + '_scribble.png')
    # color_filename = os.path.join(outdir, imname + '_color.npy')
    #  np.save(color_filename, color)
    # masked_target_filename = os.path.join(outdir, imname + '_masked_target.png')
    shape_code, color_code = latent
    if save_ply:
        with torch.no_grad():
            deep_sdf.colormesh.create_mesh(trainer.deepsdf_net, trainer.colorsdf_net, 
                    shape_code, color_code, colormesh_filename, N=256, max_batch=int(2 ** 18))
    # torch.save(latent, latent_filename) 
    pred_3D     = trainer.render_express(shape_code, color_code, resolution=512)
    pred_3D = cv2.cvtColor(pred_3D,cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_3D_filename, pred_3D) 
    # pred_rgb = trainer.render_color2d(color_code, shape_code)
    # save_image(pred_rgb, pred_rgb_filename)
    # save_image(np.moveaxis(source.squeeze().cpu().numpy(), 0, -1), source_filename)
    if target is not None:
        save_image(np.moveaxis(target.squeeze().cpu().numpy(), 0, -1), target_filename)
    save_image(np.moveaxis(scribble.squeeze().cpu().numpy(), 0, -1), scribble_filename)
    #if mask is not None:
    #    save_image(np.moveaxis((mask*target).squeeze().cpu().numpy(),0,-1),masked_target_filename)

def save_init(trainer, latent, outdir, imname):
    colormesh_filename = os.path.join(outdir, imname)
    pred_3D_filename = os.path.join(outdir, imname + '_3D.png')
    shape_code, color_code = latent
    with torch.no_grad():
        deep_sdf.colormesh.create_mesh(trainer.deepsdf_net, trainer.colorsdf_net, 
                shape_code, color_code, colormesh_filename, N=256, max_batch=int(2 ** 18))
    pred_3D     = trainer.render_express(shape_code, color_code, resolution=512)
    pred_3D = cv2.cvtColor(pred_3D,cv2.COLOR_RGB2BGR)
    cv2.imwrite(pred_3D_filename, pred_3D) 

def is_exist(outdir, imname):
    mesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + '.pth')
    pred_rgb_filename = os.path.join(outdir, imname + '_rgb.png')
    pred_3D_filename = os.path.join(outdir, imname + '_3D.png')
    target_filename = os.path.join(outdir, imname + '_target.png')
    if (os.path.exists(mesh_filename) and os.path.exists(latent_filename) and os.path.exists(pred_rgb_filename) and os.path.exists(pred_3D_filename) and os.path.exists(target_filename)):
        return True
    else:
        return False

def save_image(image, outname):
    out = np.uint8(image * 255)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outname, out)

def load_image(path, imsize=128):
    transform = transforms.Compose([transforms.Resize((imsize,imsize)), transforms.ToTensor(),])
    data_im = Image.open(path)
    x = np.array(data_im)
    if x.shape[-1] == 4:
        color = (255,255,255)
        r, g, b, a = np.rollaxis(x, axis=-1) 
        r[a == 0] = color[0]
        g[a == 0] = color[1]
        b[a == 0] = color[2]
        x = np.dstack([r, g, b])
        data_im = Image.fromarray(x, 'RGB')
    data = transform(data_im)
    return data

def reconstruct(trainer, target, mask, epoch, trial, gamma, beta):
    temp_shape, temp_color = trainer.get_known_latent(0) 
    min_loss = np.inf
    for i in range(trial):
        init_shape = torch.randn_like(temp_shape).to(temp_shape.device)
        init_color = torch.randn_like(temp_color).to(temp_color.device)
        latent, loss = trainer.step_recon_rgb(init_shape, init_color, target, mask=mask,
                                        epoch=epoch, gamma=gamma, beta=beta)
        if min_loss > loss:
            best_latent = latent[-1]
            min_loss = loss
    return best_latent

def edit(trainer, init_latent, target, mask, epoch, trial, gamma, beta):
    min_loss = np.inf
    init_shape, init_color = init_latent
    temp_shape, temp_color = trainer.get_known_latent(0) 
    since = time.time()
    for i in range(trial):
        init_color = torch.randn_like(temp_color).to(temp_color.device)
        latent, loss = trainer.step_edit_rgb(init_shape, init_color, target, mask=mask,
                                        epoch=epoch, gamma=gamma, beta=beta)
        if min_loss > loss:
            best_latent = latent[-1]
            min_loss = loss
    elapse = time.time() - since
    print(f"It takes {elapse} seconds to edit the color")
    return best_latent


def get_args():
    parser = argparse.ArgumentParser(description='Reconstruction')
    parser.add_argument('config', type=str, help='The configuration file.')
    parser.add_argument('--pretrained', default=None, type=str, help='pretrained model checkpoint')
    parser.add_argument('--outdir', default=None, type=str, help='path of output')
    parser.add_argument('--category', default="airplane", type=str, help='path of output')
    parser.add_argument('--imagelist', default=None, type=str, help='the dir of images')
    parser.add_argument('--imagename', default=None, type=str, help='the name of target image')
    parser.add_argument('--mask', default=False, action='store_true')
    parser.add_argument('--ref-only', default=False, action='store_true')
    parser.add_argument('--mask-level', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.02, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--epoch', default=1001, type=int)
    parser.add_argument('--trial', default=20, type=int)
    parser.add_argument('--partid', default=4, type=int)
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


def load_image_and_scribble(source_path, target_path, part_list, use_target=True):

    imagelist = glob.glob(os.path.join(source_path, "*Layer-*.png"))
    if len(imagelist) == 0:
        return None
    masks = []
    for part in part_list:
        for impath in imagelist:
            if f"Layer-{part}" in impath: # layer1 means the orig image
                im = cv2.resize(cv2.imread(impath), (128, 128))[:, :, 0]
                bg_value = np.max(im)
                masks.append((im != bg_value).astype(int))
                break
    if len(masks) == 0:
        return None

    # get a higher resolutional scribble for paper visualization
    masks2 = []
    for part in part_list:
        for impath in imagelist:
            if f"Layer-{part}" in impath: # layer1 means the orig image
                im = cv2.resize(cv2.imread(impath), (512, 512))[:, :, 0]
                bg_value = np.max(im)
                masks2.append((im != bg_value).astype(int))
                break

    # load source and target image
    source_image = os.path.join(source_path, "render_r_000.png")
    source_im = load_image(source_image)
    bgrs = np.random.rand(len(masks), 3) 
    if use_target: # color source is the reference image
        target_image = os.path.join(target_path, "render_r_000.png")
        target_im = load_image(target_image)
        scribble = torch.zeros_like(target_im)
        for mpart in masks:
            scribble += target_im * torch.from_numpy(mpart)
        # higher resolution
        target_im2 = load_image(target_image, imsize=512)
        scribble2 = torch.zeros_like(target_im2)
        for mpart in masks2:
            scribble2 += target_im2 * torch.from_numpy(mpart)
    else: # color source is random color
        target_im = None
        scribble = torch.zeros_like(source_im)
        C, H, W = source_im.shape
        for idx, mpart in enumerate(masks): 
            b, g, r = bgrs[idx]
            bgr = torch.tensor([b, g, r]).unsqueeze(1).unsqueeze(1).expand(3, H, W)
            scribble += bgr * torch.from_numpy(mpart)
        
        scribble2 = torch.zeros(C, 512, 512)
        for idx, mpart in enumerate(masks2): 
            b, g, r = bgrs[idx]
            bgr2 = torch.tensor([b, g, r]).unsqueeze(1).unsqueeze(1).expand(3, 512, 512)
            scribble2 += bgr2 * torch.from_numpy(mpart)
        
    mask_im = torch.from_numpy((np.sum(np.array(masks), axis=0) > 0).astype(int))

    return {"source": source_im, "target": target_im, 
            "scribble": scribble, "mask":mask_im, "scribble2": scribble2, "color": bgrs}


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
    source_dir = os.path.join(args.imagelist, 'source')
    target_dir = os.path.join(args.imagelist, 'target')
    def getname(impath):
        return impath.split("/")[-2]
    os.makedirs(args.outdir, exist_ok=True)

    # part_list: the id indicates the semantic parts 
    if args.category == "airplane":
        if args.partid == 0:
            part_list = [2, 3] # body and left wing 
        elif args.partid == 1:
            part_list = [2, 3, 4] # body
        elif args.partid == 2:
            part_list = [3] # left wing 

    elif args.category  == "chair":
        if args.partid == 0:
            part_list= [2, 3, 4, 6] # seat, back, left leg, left arm
        elif args.partid == 1:
            part_list = [2, 3] # seat, back
        elif args.partid == 2:
            part_list = [2, 4] # seat, left leg
        elif args.partid == 3:
            part_list = [2, 6] # seat, left arm
        elif args.partid == 4:
            part_list = [2] # seat
    else:
        print("No such category")
        exit()

    # edit known shapes (i.e. shapes from the training dataset)
    imname = args.imagename
    source_path = os.path.join(source_dir, imname)
    targetdir = os.path.join(args.outdir, imname)
    os.makedirs(targetdir, exist_ok=True)
    target_path = os.path.join(target_dir, imname)
    print("Edit 3D from %s ..." % imname)
    source_latent = trainer.get_known_latent(trainer.sid2idx[imname])

    # save init
    initdir = os.path.join(targetdir, "init")
    os.makedirs(initdir, exist_ok=True)
    save_init(trainer, source_latent, initdir, imname)

    # randomize the color of scribbles
    randdir = os.path.join(targetdir, "rand")
    os.makedirs(randdir, exist_ok=True)
    for k in range(30): 
        data = load_image_and_scribble(source_path, target_path, part_list, use_target=False)
        source_latent = trainer.get_known_latent(trainer.sid2idx[imname])
        edit_latent = edit(trainer, source_latent, data["scribble"], data["mask"], 
                            args.epoch, args.trial, args.gamma, args.beta)
        save(trainer, edit_latent, data["source"], data["target"], 
             data["scribble2"], data["mask"], data["color"], 
             randdir, imname+f"_{k}", save_ply=True)

if __name__ == "__main__":
    args, cfg = get_args()
    main(args, cfg)

