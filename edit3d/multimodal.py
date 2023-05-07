import os

from edit3d.PinMemDict import PinMemDict
from edit3d.loaders.NPYLoaderN import NPYLoaderN
from edit3d.samplers.SequentialWarpSampler import SequentialWarpSampler
from edit3d.samplers.ShuffleWarpSampler import ShuffleWarpSampler

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

torch.set_num_threads(1)
import numpy as np
from torch.utils.data import Dataset
import json
import logging

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=logging.getLevelName(log_level))

logger = logging.getLogger(__name__)
CUDA_DEVICE = "cuda:0"

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() and os.getenv("USE_GPU") else "cpu")


def init_np_seed():
    torch.set_num_threads(1)
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)  # numpy seed must be between 0 and 2**32-1


def np_collate(batch):
    batch_z = zip(*batch)
    return [torch.stack([torch.from_numpy(b) for b in batch_z_z], 0) for batch_z_z in batch_z]


def np_collate_dict(batch):
    b_out = {}
    for k in batch[0].keys():
        c = []
        for b in batch:
            c.append(b[k])
        if type(c[0]) is np.ndarray:
            c = torch.from_numpy(np.stack(c, axis=0))
        elif type(c[0]) is torch.Tensor:
            c = torch.stack(c, dim=0)
        else:
            pass
        b_out[k] = c
    return PinMemDict(b_out)


def get_data_loaders(args):
    # Load split file
    with open(args.split_files.train) as split_data:
        sp = json.load(split_data)
        train_split = set(sp["ShapeNetV2"][args.cate_id])

    with open(args.split_files.test) as split_data:
        sp = json.load(split_data)
        test_split = set(sp["ShapeNetV2"][args.cate_id])

    train_data_list = []
    test_data_list = []
    sphere_list = set()
    with os.scandir(args.sdf_data_dir.sphere) as npy_list:
        for npy_path in npy_list:
            if npy_path.is_file():
                sphere_list.add(npy_path.name.split(".")[0])

    # load image data
    im_path = {}
    with os.scandir(args.sdf_data_dir.sketch) as im_list:
        for im_name in im_list:
            if im_name.name in sphere_list:
                # im_path[im_name.name] = os.path.join(args.sdf_data_dir.sketch, im_name.name, "sketch-F-2.png")
                im_path[im_name.name] = os.path.join(args.sdf_data_dir.sketch, im_name.name, args.sketch_name)

    # load color data
    color2d_path = {}
    with os.scandir(args.sdf_data_dir.color) as im_list:
        for im_name in im_list:
            if im_name.name in sphere_list:
                color2d_path[im_name.name] = os.path.join(args.sdf_data_dir.color, im_name.name, "render_r_000.png")

    with os.scandir(args.sdf_data_dir.surface) as npy_list:
        for npy_path in npy_list:
            if npy_path.is_file():
                shape_id = npy_path.name.split(".")[0]
                if (shape_id in sphere_list) and (shape_id in color2d_path.keys()) and (shape_id in im_path.keys()):
                    surface_path = npy_path.path
                    sphere_path = os.path.join(args.sdf_data_dir.sphere, npy_path.name)
                    if shape_id in train_split:
                        train_data_list.append(
                            (
                                shape_id,
                                surface_path,
                                sphere_path,
                                im_path[shape_id],
                                color2d_path[shape_id],
                            )
                        )
                    if shape_id in test_split:
                        test_data_list.append(
                            (
                                shape_id,
                                surface_path,
                                sphere_path,
                                im_path[shape_id],
                                color2d_path[shape_id],
                            )
                        )
                else:
                    logger.error(f"ERROR! {shape_id} not found in coarse SDFs.")

    train_data_list.sort()
    test_data_list.sort()
    logger.info("[get_data_loaders] #train: %s; #test: %s.", len(train_data_list), len(test_data_list))

    train_dataset = NPYLoaderN(
        train_data_list,
        args.train.num_sample_points.fine,
        args.train.num_sample_points.coarse,
        imsize=args.train.imsize,
    )
    train_sampler = ShuffleWarpSampler(train_dataset, n_repeats=args.train.num_repeats)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.train.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=np_collate_dict,
        worker_init_fn=init_np_seed,
    )

    if getattr(args.test, "test_on_train_set", False):
        test_data_list = train_data_list
        logger.info("[NewSDF Dataset] Testing on train set...")
    if getattr(args.test, "subset", None):
        test_data_list = test_data_list[: args.test.subset]
        logger.info("[get_data_loaders] Subsetting test set to {}".format(args.test.subset))
    test_dataset = NPYLoaderN(
        test_data_list,
        args.test.num_sample_points.fine,
        args.test.num_sample_points.coarse,
        imsize=args.test.imsize,
    )
    test_sampler = SequentialWarpSampler(test_dataset, n_repeats=args.test.num_repeats)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.test.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=np_collate_dict,
        worker_init_fn=init_np_seed,
    )

    train_shape_ids = [x[0] for x in train_data_list]
    test_shape_ids = [x[0] for x in test_data_list]

    loaders = {
        "train_loader": train_loader,
        "train_shape_ids": train_shape_ids,
        "test_loader": test_loader,
        "test_shape_ids": test_shape_ids,
    }
    return loaders
