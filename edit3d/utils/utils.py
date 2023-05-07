import argparse
import os
import yaml

def exists(outdir: str, imname: str):
    mesh_filename = os.path.join(outdir, imname)
    latent_filename = os.path.join(outdir, imname + ".pth")
    pred_rgb_filename = os.path.join(outdir, imname + "_rgb.png")
    pred_3d_filename = os.path.join(outdir, imname + "_3D.png")
    target_filename = os.path.join(outdir, imname + "_target.png")
    if (
            os.path.exists(mesh_filename)
            and os.path.exists(latent_filename)
            and os.path.exists(pred_rgb_filename)
            and os.path.exists(pred_3d_filename)
            and os.path.exists(target_filename)
    ):
        return True
    else:
        return False


def dict2namespace(config: dict):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
