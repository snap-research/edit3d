import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms

import logging

logger = logging.getLogger(__name__)

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
        logger.info(
            "[NPYLoaderN] Number of shapes: {}; #fine: {}; #coarse: {}.".format(
                len(filelist), npoints_fine, npoints_coarse
            )
        )

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        shape_id, surface_file, sphere_file, sketch_file, color2d_file = self.filelist[idx]

        # for surface samples: [xyzd_inside; xyzd_outside] half inside, half outside
        num_inside_points = int(self.npoints_fine * 0.45)
        num_outside_points = num_inside_points
        num_sphere_points = self.npoints_fine - num_inside_points - num_outside_points
        # Surface samples
        data = np.load(surface_file, mmap_mode="r")
        num_samples = data.shape[0]
        data_sur = np.array([[]])
        if num_samples:
            subset_idx_inside = np.random.choice(num_samples // 2, num_inside_points, replace=True)
            subset_idx_outside = np.random.choice(num_samples // 2, num_outside_points, replace=True) + num_samples // 2
            subset_idx = np.concatenate([subset_idx_inside, subset_idx_outside])
            data_sur = data[subset_idx, :]  # xyzd + RGB
        # Sphere samples
        data = np.load(sphere_file, mmap_mode="r")
        num_samples = data.shape[0]
        data_sph = np.array([[]])
        if num_samples:
            num_samples = data.shape[0]
            subset_idx = np.random.choice(num_samples, num_sphere_points, replace=True)
            data_sph = data[subset_idx, :]  # xyzd + RGB (-1, -1, -1)
        # combine
        data_f = np.concatenate([data_sur, data_sph], axis=0)

        # sketch samples
        data_im = Image.open(sketch_file)
        data_im = self.transform(data_im)  # N*C*H*W
        data_im = data_im.mean(dim=0).round().unsqueeze(0)  # C is binary black or white.

        # color image samples
        data_color = Image.open(color2d_file)
        data_color = self.transform(data_color)[0:3]  # N*C*H*W  no alpha channel

        idx_t = np.array([idx], dtype=np.longlong)

        # the data_f contains xyz+d+rgb
        return {
            "surface_samples": data_f,
            # "sphere_samples": np.array([]),
            "sketch": data_im,
            "color_2d": data_color,
            "shape_indices": idx_t,
            "shape_ids": shape_id,
        }
