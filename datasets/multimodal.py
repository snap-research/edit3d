import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch

torch.set_num_threads(1)
import numpy as np
from torch.utils.data import Dataset, Sampler
from torch.utils import data
import random
import time
import itertools
import json
from PIL import Image
import torchvision.transforms as transforms


def init_np_seed(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)  # numpy seed must be between 0 and 2**32-1


def np_collate(batch):
    batch_z = zip(*batch)
    return [
        torch.stack([torch.from_numpy(b) for b in batch_z_z], 0)
        for batch_z_z in batch_z
    ]


# czz: the pinned memory is just used to speed up the transfer between CPU and GPU
# because the data is loaded on CPU first.
class PinMemDict:
    def __init__(self, data):
        self.data = data

    # custom memory pinning method on custom type
    def pin_memory(self):
        out_b = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                out_b[k] = v.pin_memory()
            else:
                out_b[k] = v
        return out_b


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


class ShuffleWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        self.data_source = data_source
        self.n_repeats = n_repeats
        print("[ShuffleWarpSampler] Expanded data size: {}".format(len(self)))

    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = torch.randperm(len(self.data_source)).tolist()
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)

    def __len__(self):
        return len(self.data_source) * self.n_repeats


class SequentialWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        self.data_source = data_source
        self.n_repeats = n_repeats
        print("[SequentialWarpSampler] Expanded data size: {}".format(len(self)))

    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = list(range(len(self.data_source)))
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)

    def __len__(self):
        return len(self.data_source) * self.n_repeats


class NPYLoaderN(Dataset):
    def __init__(
        self,
        filelist,
        npoints_fine=2048,
        npoints_coarse=2048,
        only_sketch=False,
        imsize=64,
    ):
        self.filelist = filelist
        self.npoints_fine = npoints_fine
        self.npoints_coarse = npoints_coarse
        self.transform = transforms.Compose(
            [
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
            ]
        )
        self.only_sketch = only_sketch
        print(
            "[NPYLoaderN] Number of shapes: {}; #fine: {}; #coarse: {}.".format(
                len(filelist), npoints_fine, npoints_coarse
            )
        )

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        shape_id, surface_file, sphere_file, sketch_file, color2d_file = self.filelist[
            idx
        ]

        # for surface samples: [xyzd_inside; xyzd_outside] half inside, half outside
        num_inside_points = int(self.npoints_fine * 0.45)
        num_outside_points = num_inside_points
        num_sphere_points = self.npoints_fine - num_inside_points - num_outside_points
        # Surface samples
        data = np.load(surface_file, mmap_mode="r")
        num_samples = data.shape[0]
        subset_idx_inside = np.random.choice(
            num_samples // 2, num_inside_points, replace=True
        )
        subset_idx_outside = (
            np.random.choice(num_samples // 2, num_outside_points, replace=True)
            + num_samples // 2
        )
        subset_idx = np.concatenate([subset_idx_inside, subset_idx_outside])
        data_sur = data[subset_idx, :]  # xyzd + RGB
        # Sphere samples
        data = np.load(sphere_file, mmap_mode="r")
        num_samples = data.shape[0]
        subset_idx = np.random.choice(num_samples, num_sphere_points, replace=True)
        data_sph = data[subset_idx, :]  # xyzd + RGB (-1, -1, -1)
        # combine
        data_f = np.concatenate([data_sur, data_sph], axis=0)
        data_c = np.array([])

        # sketch samples
        data_im = Image.open(sketch_file)
        data_im = self.transform(data_im)  # N*C*H*W

        # color image samples
        data_color = Image.open(color2d_file)
        data_color = self.transform(data_color)  # N*C*H*W

        idx_t = np.array([idx], dtype=np.long)

        # the data_f contains xyz+d+rgb
        return {
            "surface_samples": data_f,
            "sphere_samples": data_c,
            "sketch": data_im,
            "color_2d": data_color,
            "shape_indices": idx_t,
            "shape_ids": shape_id,
        }


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
                im_path[im_name.name] = os.path.join(
                    args.sdf_data_dir.sketch, im_name.name, args.sketch_name
                )

    # load color data
    color2d_path = {}
    with os.scandir(args.sdf_data_dir.color) as im_list:
        for im_name in im_list:
            if im_name.name in sphere_list:
                color2d_path[im_name.name] = os.path.join(
                    args.sdf_data_dir.color, im_name.name, "render_r_000.png"
                )

    with os.scandir(args.sdf_data_dir.surface) as npy_list:
        for npy_path in npy_list:
            if npy_path.is_file():
                shape_id = npy_path.name.split(".")[0]
                if (
                    (shape_id in sphere_list)
                    and (shape_id in color2d_path.keys())
                    and (shape_id in im_path.keys())
                ):
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
                    print("ERROR! {} not found in coarse SDFs.".format(shape_id))

    train_data_list.sort()
    test_data_list.sort()
    print(
        "[get_data_loaders] #train: {}; #test: {}.".format(
            len(train_data_list), len(test_data_list)
        )
    )

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
        print("[NewSDF Dataset] Testing on train set...")
    if getattr(args.test, "subset", None):
        test_data_list = test_data_list[: args.test.subset]
        print("[get_data_loaders] Subsetting test set to {}".format(args.test.subset))
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


if __name__ == "__main__":
    pass
