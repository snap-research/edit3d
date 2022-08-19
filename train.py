import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import yaml
from tensorboardX import SummaryWriter
from shutil import copy2
import sys
import json
import importlib
import torchvision

def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='MultiModalVAD Training')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    
    parser.add_argument('--pretrained', default=None, type=str,
                        help='pretrained model checkpoint')
                        
    # For easy debugging:
    parser.add_argument('--test_run', default=False, action='store_true')
    
    parser.add_argument('--special', default=None, type=str,
                        help='Run special tasks')
                        
    # output path
    parser.add_argument('--logdir', default=None, type=str,
                        help='log path')


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

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    config = dict2namespace(config)

    #  Create log_name
    log_prefix = ''
    if args.test_run:
        log_prefix = 'tmp_'
    if args.special is not None:
        log_prefix = log_prefix + 'special_{}_'.format(args.special)
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # Currently save dir and log_dir are the same
    config.log_name = "{}/{}{}_{}".format(args.logdir, log_prefix, cfg_file_name, run_time)
    config.save_dir = "{}/{}{}_{}/checkpoints".format(args.logdir, log_prefix, cfg_file_name, run_time)
    config.log_dir = "{}/{}{}_{}".format(args.logdir, log_prefix, cfg_file_name, run_time)
    os.makedirs(os.path.join(config.log_dir, 'config'))
    os.makedirs(config.save_dir)
    copy2(args.config, os.path.join(config.log_dir, 'config'))
    with open(os.path.join(config.log_dir, 'config', 'argv.json'), 'w') as f:
        json.dump(sys.argv, f)
    return args, config

def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(cfg.log_name)
    device = torch.device('cuda:0')
    
    # Load experimental settings
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    train_shape_ids = loaders['train_shape_ids']
    cfg.train_shape_ids = train_shape_ids
    test_loader = loaders['test_loader']
    test_shape_ids = loaders['test_shape_ids']
    cfg.test_shape_ids = test_shape_ids

    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args, device)
    
    # Prepare for training
    start_epoch = 0
    trainer.prep_train()
    
    if args.resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained)
        else:
            start_epoch = trainer.resume(cfg.resume.dir)
    
    if args.special is not None:
        special_fun = getattr(trainer, args.special)
        special_fun(test_loader=test_loader, writer=writer)
        exit()
    
    # Main training loop
    prev_time = time.time()
    print("[Train] Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step_cnt = 0
    for epoch in range(start_epoch, cfg.trainer.epochs):

        trainer.epoch_start(epoch)
        # train for one epoch
        prev_load_data = time.time()
        for bidx, data in enumerate(train_loader):
            step_cnt = bidx + len(train_loader) * epoch + 1
            
            # print("load data time: {:0.5f}".format(time.time() - prev_load_data))
            logs_info = trainer.step(data)
            
            # Print info
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = time.time()
            print('Epoch: {}; Iter: {}; Time: {:0.5f}; Loss_shape: {:f}; Loss_color3D: {:f}; Loss_sketch: {:f}; Loss_color2D: {:f},'.format(epoch, bidx, elapsed_time, logs_info["loss_shape"], logs_info["loss_color3D"] * 0.1, logs_info["loss_sketch"], logs_info["loss_color2D"]))
            if "loss_shape_latent" in logs_info.keys():
                print('Epoch: {}; Iter: {}; Time: {:0.5f}; Loss_shape: {:f}; Loss_color3D: {:f}; Loss_sketch: {:f}; Loss_color2D: {:f}, Loss_shape_latent: {:f}, Loss_color_latent: {:f},'.format(epoch, bidx, elapsed_time, logs_info["loss_shape"], logs_info["loss_color3D"] * 0.1, logs_info["loss_sketch"], logs_info["loss_color2D"], logs_info["loss_shape_latent"], logs_info["loss_color_latent"]))
            
            # Log
            if step_cnt % int(cfg.viz.log_interval) == 0:
                if writer is not None:
                    for k, v in logs_info.items():
                        writer.add_scalar(k, v, step_cnt)
            prev_load_data = time.time()
            
        # Save checkpoints
        if (epoch + 1) % int(cfg.viz.save_interval) == 0:
            
            # visualize the generated sketchs
            im_data = trainer.sample_images(data)
            grid_gt = torchvision.utils.make_grid(im_data["gt_sketch"])
            grid_sample = torchvision.utils.make_grid(im_data["gen_sketch"])
            writer.add_image("sketch_gts", grid_gt, step_cnt)
            writer.add_image("sketch_samples", grid_sample, step_cnt)

            grid_gt = torchvision.utils.make_grid(im_data["gt_color"])
            grid_sample = torchvision.utils.make_grid(im_data["gen_color"])
            writer.add_image("color_gts", grid_gt, step_cnt)
            writer.add_image("color_samples", grid_sample, step_cnt)

            # visualize the generated sdf
            im_data = trainer.sample_images(data)
            renders = torch.from_numpy(np.array(im_data["render_sdf"])).permute(0, 3, 1, 2) 
            grid_sample = torchvision.utils.make_grid(renders)
            writer.add_image("render_samples", grid_sample, step_cnt)

            # checkpoints     
            trainer.save(epoch=epoch, step=step_cnt)

        
        trainer.epoch_end(epoch, writer=writer)
        # writer.flush()
    
    # always save last epoch
    if (epoch + 1) % int(cfg.viz.save_interval) != 0:
        trainer.save(epoch=epoch, step=step_cnt)
    writer.close()
            
    
if __name__ == "__main__":
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main(args, cfg)
    
    
