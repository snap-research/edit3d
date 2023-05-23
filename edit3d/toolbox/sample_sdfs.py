## Updated implementation based on https://github.com/zekunhao1995/DualSDF

import argparse
import json
import os
import time
import logging

# PyTorch must come before mesh2sdf
import torch  # noqa

import mesh2sdf
import numpy
import numpy as np
import pyassimp

from edit3d.toolbox import pcl_library

logger = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def sdfmeshfun(point, mesh):
    out_ker = mesh2sdf.mesh2sdf_gpu(point.contiguous(), mesh)[0]
    return out_ker


def trimmesh(mesh_t, residual=False):
    mesh_t = mesh_t.to("cuda:0")
    valid_triangles = mesh2sdf.trimmesh_gpu(mesh_t)
    if residual:
        valid_triangles = ~valid_triangles
    mesh_t = mesh_t[valid_triangles, :, :].contiguous()
    logger.info("[Trimmesh] {} -> {}".format(valid_triangles.size(0), mesh_t.size(0)))
    return mesh_t


def meshpreprocess_bsphere(mesh: numpy.array):
    mesh[:, :, 1] *= -1
    # normalize mesh
    mesh = mesh.reshape(-1, 3)
    mesh_max = np.amax(mesh, axis=0)
    mesh_min = np.amin(mesh, axis=0)
    mesh_center = (mesh_max + mesh_min) / 2
    mesh = mesh - mesh_center
    # Find the max distance to origin
    max_dist = np.sqrt(np.max(np.sum(mesh ** 2, axis=-1)))
    mesh_scale = 1.0 / max_dist
    mesh *= mesh_scale
    mesh = mesh.reshape(-1, 3, 3)
    mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
    return mesh_t


def normalize(x):
    x /= torch.sqrt(torch.sum(x ** 2))
    return x


def loader(file_path):
    with pyassimp.load(file_path) as object_file:
        logger.info(f"Loading mesh: {file_path}")
        mesh = object_file.meshes[0]
        return mesh.vertices[mesh.faces]


def process_mesh(classid, shapeid, mesh, colors=None):
    num_surface_samples = 32000
    num_sphere_samples = 32000
    target_samples = 25000

    noise_vec = torch.empty([num_surface_samples, 3], dtype=torch.float32, device=device)  # x y z
    noise_vec2 = torch.empty([num_sphere_samples, 3], dtype=torch.float32, device=device)  # x y z
    noise_vec3 = torch.empty([num_sphere_samples, 1], dtype=torch.float32, device=device)  # x y z

    logger.debug("Processing %s, %s", classid, shapeid)

    start = time.time()

    color_samples = None
    try:
        mesh = meshpreprocess_bsphere(mesh).to(device)
        if not args.notrim:
            # Remove inside triangles
            mesh = trimmesh(mesh)
        if colors.any:
            pcl, color_samples = pcl_library.mesh2pcl_color(mesh.cpu().numpy(), colors, num_surface_samples)
            pcl = torch.from_numpy(pcl).to(device)
            color_samples = torch.from_numpy(color_samples).to(device)
        else:
            pcl = torch.from_numpy(pcl_library.mesh2pcl(mesh.cpu().numpy(), colors, num_surface_samples)).to(device)
    except Exception as e:
        write_failed_id(args.output_path, shapeid)
        raise

    # Surface points
    noise_vec.normal_(0, np.sqrt(0.005))
    points1 = pcl + noise_vec
    noise_vec.normal_(0, np.sqrt(0.0005))
    points2 = pcl + noise_vec
    # Unit sphere points
    noise_vec2.normal_(0, 1)
    shell_points = noise_vec2 / torch.sqrt(torch.sum(noise_vec2 ** 2, dim=-1, keepdim=True))
    noise_vec3.uniform_(0, 1)  # r = 1
    points3 = shell_points * (noise_vec3 ** (1 / 3))
    all_points = torch.cat([points1, points2, points3], dim=0)
    all_colors = torch.cat([color_samples, torch.full(((points2.shape[0]+points3.shape[0]),3), -1, device=device)], dim=0)

    # logger.info(all_points.shape)
    sample_dist = sdfmeshfun(all_points, mesh)
    # logger.info(sample_dist.shape)

    xyzd = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1)
    xyzd = torch.cat((xyzd, all_colors), dim=1).cpu().numpy()

    xyzd_sur = xyzd[:num_surface_samples * 2]
    xyzd_sph = xyzd[num_surface_samples * 2:]

    inside_mask = (xyzd_sur[:, 3] <= 0)
    outside_mask = np.logical_not(inside_mask)

    inside_cnt = np.count_nonzero(inside_mask)
    outside_cnt = np.count_nonzero(outside_mask)
    inside_stor = [xyzd_sur[inside_mask, :]]
    outside_stor = [xyzd_sur[outside_mask, :]]
    n_attempts = 0
    while (inside_cnt < target_samples) or (outside_cnt < target_samples):
        # noise_vec.normal_(0, np.sqrt(0.005))
        # points1 = pcl + noise_vec
        # noise_vec.normal_(0, np.sqrt(0.0005))
        # points2 = pcl + noise_vec
        # all_points = torch.cat([points1, points2], dim=0)
        sample_dist = sdfmeshfun(all_points, mesh)
        xyzd_sur = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1)
        xyzd_sur = torch.cat((xyzd_sur, all_colors), dim=1).cpu().numpy()
        inside_mask = (xyzd_sur[:, 3] <= 0)
        outside_mask = np.logical_not(inside_mask)
        inside_cnt += np.count_nonzero(inside_mask)
        outside_cnt += np.count_nonzero(outside_mask)
        inside_stor.append(xyzd_sur[inside_mask, :])
        outside_stor.append(xyzd_sur[outside_mask, :])
        n_attempts += 1
        logger.debug(f" - {n_attempts}nd Attempt: {inside_cnt} / {target_samples}")
        if n_attempts > 200 or ((np.minimum(inside_cnt, outside_cnt) / n_attempts) < 500):
            with open(f'bads_list_{classid}.txt', 'a+') as f:
                f.write(f'{classid},{shapeid},{np.minimum(inside_cnt, outside_cnt)},{n_attempts}\n')
            break

    xyzd_inside = np.concatenate(inside_stor, axis=0)
    xyzd_outside = np.concatenate(outside_stor, axis=0)

    num_yields = np.minimum(xyzd_inside.shape[0], xyzd_outside.shape[0])
    xyzd_inside = xyzd_inside[:num_yields, :]
    xyzd_outside = xyzd_outside[:num_yields, :]

    xyzd = np.concatenate([xyzd_inside, xyzd_outside], axis=0)
    end = time.time()
    logger.info(f"[Perf] time: {end - start}, yield: {num_yields}")
    return xyzd, xyzd_sph


def get_mesh_loader(split_file, data_dir):
    """
    Returns a lazy loaded mesh (i.e. mesh = loader()), the current id, next id, and percent complete.
    """
    completed = 0
    if split_file:
        with open(split_file, 'r') as open_split_file:
            splits = json.load(open_split_file)
            total = 0
            for dataset, l1 in splits.items():
                if dataset.lower() == 'shapenetv2':
                    total += sum(len(ids) for ids in l1.values())
                    for l1_name, l2 in l1.items():
                        for idx, l2_name in enumerate(l2):
                            file_path = os.path.join(data_dir, dataset, l1_name, l2_name, "models",
                                                     "model_normalized.obj")
                            try:
                                next_id = l2[idx + 1]
                            except IndexError:
                                next_id = None
                            completed += 1
                            yield (lambda: loader(file_path)), l2_name, next_id, completed / total
                else:
                    raise Exception("Split file dataset not supported yet.")
    else:
        npy_list = list(filter(
            lambda np_path: all([np_path.is_file(), np_path.path.endswith('.polygons.npy')]),
            os.scandir(data_dir)))
        total = len(npy_list)
        for idx, npy_path in enumerate(npy_list):
            logger.info(f"Loading mesh: {npy_path.path}")
            current_id = os.path.splitext(os.path.basename(npy_path.path))[0]
            try:
                next_id = os.path.splitext(os.path.basename(npy_list[idx + 1]))[0]
            except IndexError:
                next_id = None
            completed += 1
            yield (lambda: (
            numpy.load(npy_path.path), numpy.load(f"{npy_path.path.removesuffix('.polygons.npy')}.color.npy"))), current_id, next_id, completed / total


def write_failed_id(path, id):
    with open(os.path.join(path, "failures.json"), 'w+') as failed_file:
        failed_id_text = failed_file.read()
        failed_ids = json.loads(failed_id_text) if failed_id_text else []
        failed_ids.append(id)
        failed_file.write(json.dumps(failed_ids))


def main(args):
    # for each mesh, sample points within bounding sphere.
    # According to DeepSDF paper, 250,000x2 points around the surface,
    # 25,000 points within the unit sphere uniformly
    # To sample points around the surface, 
    #   - sample points uniformly on the surface,
    #   - Perturb the points with gaussian noise var=0.0025 and 0.00025
    #   - Then compute SDF

    surface_path = os.path.join(args.output_path, args.class_name + "_surface")
    sphere_path = os.path.join(args.output_path, args.class_name + "_sphere")
    for output_path in [surface_path, sphere_path]:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    for loader, shapeid, next_shapeid, progress in get_mesh_loader(args.split_file, args.mesh_npy_path):
        # If this path and the NEXT path already exists assume this one finished successfully and move on.
        if next_shapeid and all([os.path.exists(os.path.join(surface_path, f'{shapeid}.npy')),
                                 os.path.exists(os.path.join(sphere_path, f'{shapeid}.npy')),
                                 os.path.exists(os.path.join(surface_path, f'{next_shapeid}.npy')),
                                 os.path.exists(os.path.join(sphere_path, f'{next_shapeid}.npy'))]):
            logger.info("ID %s already exists.", shapeid)
            continue

        if not shapeid:
            continue

        classid = args.class_name
        try:
            xyzd, xyzd_sph = process_mesh(classid, shapeid, *loader())
            np.save(os.path.join(surface_path, f'{shapeid}.npy'), xyzd)
            np.save(os.path.join(sphere_path, f'{shapeid}.npy'), xyzd_sph)
        except Exception:
            logger.error("Failed to write %s", shapeid, exc_info=True)
        logger.info(f"Completed: {progress * 100}%")


if __name__ == "__main__":
    # python edit3d/toolbox/sample_sdfs.py chairs datasets/chairs/chairs_meshes datasets/chairs
    parser = argparse.ArgumentParser(
        description='Sample SDF values from meshes. All the NPY files under mesh_npy_path and its child dirs will be converted and the directory structure will be preserved.')
    parser.add_argument('class_name', type=str,
                        help='A name for the category of shapes being sampled (e.g. chairs)')
    parser.add_argument('mesh_npy_path', type=str,
                        help='The dir containing meshes in NPY format [ #triangles x 3(vertices) x 3(xyz) ]')
    parser.add_argument('output_path', type=str,
                        help='The output dir containing sampled SDF in NPY format [ #points x 4(xyzd) ]')
    parser.add_argument('--split_file', type=str,
                        help='The directory to the file used to look up shapes.')
    parser.add_argument('--notrim', default=False, action='store_true')
    # parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    main(args)
