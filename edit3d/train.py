import argparse
import importlib
import json
import os
import sys
import time
from shutil import copy2

import numpy as np
import torch
import torchvision
import yaml
from tensorboardX import SummaryWriter

from edit3d import device
from edit3d.utils.utils import dict2namespace
import logging

logger = logging.getLogger(__name__)

def main(args, cfg):
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(cfg.log_name)

    # Load experimental settings
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data)
    train_loader = loaders["train_loader"]
    train_shape_ids = loaders["train_shape_ids"]
    cfg.train_shape_ids = train_shape_ids
    test_loader = loaders["test_loader"]
    test_shape_ids = loaders["test_shape_ids"]
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
        logger.warning(f"Running special fun {args.special} and then exiting.")
        special_fun = getattr(trainer, args.special)
        special_fun(test_loader=test_loader, writer=writer)
        exit()

    # Main training loop
    prev_time = time.time()
    logger.info("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step_cnt = 0
    for epoch in range(start_epoch, cfg.trainer.epochs):
        logger.debug("Epoch: %d" % epoch)
        trainer.epoch_start(epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):

            step_cnt = bidx + len(train_loader) * epoch + 1

            # print("load data time: {:0.5f}".format(time.time() - prev_load_data))
            logs_info = trainer.step(data)

            # Print info
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = time.time()
            logger.info(
                "Epoch: {}; Iter: {}; Time: {:0.5f}; Loss_shape: {:f}; Loss_color3D: {:f}; Loss_sketch: {:f}; Loss_color2D: {:f},".format(
                    epoch,
                    bidx,
                    elapsed_time,
                    logs_info["loss_shape"],
                    logs_info["loss_color3D"] * 0.1,
                    logs_info["loss_sketch"],
                    logs_info["loss_color2D"],
                )
            )
            if "loss_shape_latent" in logs_info.keys():
                logger.info(
                    "Epoch: {}; Iter: {}; Time: {:0.5f}; Loss_shape: {:f}; Loss_color3D: {:f}; Loss_sketch: {:f}; Loss_color2D: {:f}, Loss_shape_latent: {:f}, Loss_color_latent: {:f},".format(
                        epoch,
                        bidx,
                        elapsed_time,
                        logs_info["loss_shape"],
                        logs_info["loss_color3D"] * 0.1,
                        logs_info["loss_sketch"],
                        logs_info["loss_color2D"],
                        logs_info["loss_shape_latent"],
                        logs_info["loss_color_latent"],
                    )
                )

            # Log
            if step_cnt % int(cfg.viz.log_interval) == 0:
                if writer is not None:
                    for k, v in logs_info.items():
                        writer.add_scalar(k, v, step_cnt)

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

        # always save last epoch
        if (epoch + 1) % int(cfg.viz.save_interval) != 0:
            trainer.save(epoch=epoch, step=step_cnt)
        writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MultiModalVAD Training")
    parser.add_argument("config", type=str, help="The configuration file.")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--pretrained", default=None, type=str, help="pretrained model checkpoint")
    parser.add_argument("--test_run", default=False, action="store_true")
    parser.add_argument("--special", default=None, type=str, help="Run special tasks")
    parser.add_argument("--logdir", default=None, type=str, help="log path")

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    #  Create log_name
    log_prefix = ""
    if args.test_run:
        log_prefix = "tmp_"
    if args.special is not None:
        log_prefix = f"{log_prefix}_special_{args.special}_"
    cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]
    run_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    # Currently save dir and log_dir are the same
    config.log_name = f"{args.logdir}/{log_prefix}{cfg_file_name}_{run_time}"
    config.save_dir = f"{config.log_name}/checkpoints"
    config.log_dir = config.log_name
    config_dir = os.path.join(config.log_dir, "config")
    os.makedirs(config_dir)
    os.makedirs(config.save_dir)
    copy2(args.config, config_dir)
    with open(os.path.join(config_dir, "argv.json"), "w") as f:
        json.dump(sys.argv, f)

    main(args, config)
