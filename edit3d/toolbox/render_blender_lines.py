# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Tested with Blender 2.9
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os, math, re
import bpy
import json
from glob import glob


def setup(args):
    # Set up rendering
    context = bpy.context
    render = bpy.context.scene.render

    render.engine = args.engine
    render.image_settings.color_mode = "RGBA"  # ('RGB', 'RGBA', ...)
    render.image_settings.color_depth = args.color_depth  # ('8', '16')
    render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    render.resolution_x = args.resolution
    render.resolution_y = args.resolution
    render.resolution_percentage = 100
    render.film_transparent = True

    bpy.context.scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    nodes.new("CompositorNodeRLayers")

    # Delete default cube
    context.active_object.select_set(True)
    bpy.ops.object.delete()


def import_obj(file_path, scale=1, remove_doubles=True, edge_split=True, **kwargs):
    bpy.ops.object.select_all(action="DESELECT")
    match os.path.splitext(file_path)[1]:
        case ".obj":
            bpy.ops.import_scene.obj(filepath=file_path)
        case ".ply":
            bpy.ops.import_mesh.ply(filepath=file_path)
        case _:
            raise Exception(f"The {os.path.splitext(file_path)[1]} is not yet supported.")
    obj = set_active_object()
    obj.pass_index = 1

    if scale != 1:
        # Probably can just move the camera instead of resizing the object.
        bpy.ops.transform.resize(value=(scale, scale, scale))
        bpy.ops.object.transform_apply(scale=True)
    if remove_doubles:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode="OBJECT")
    if edge_split:
        bpy.ops.object.modifier_add(type="EDGE_SPLIT")
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(modifier="EdgeSplit")
    return obj


def setup_camera() -> (bpy.types.Camera, bpy.types.Camera):
    # Place camera
    cam = bpy.context.scene.objects["Camera"]
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    cam_target = bpy.data.objects.new("Empty", None)
    cam_target.location = (0, 0, 0)
    cam.parent = cam_target

    bpy.context.scene.collection.objects.link(cam_target)
    set_active_object(cam_target)
    cam_constraint.target = cam_target
    return cam, cam_target


def add_line_art(obj):
    set_active_object(obj)

    try:
        gpencil = bpy.context.scene.objects['GPencil']
        gpencil.grease_pencil_modifiers["Line Art"].source_object = obj
    except KeyError:
        # Add stroke materials
        bpy.ops.object.gpencil_add(type='EMPTY')
        bpy.ops.object.gpencil_modifier_add(type='GP_LINEART')
        gpencil = bpy.context.scene.objects['GPencil']
        gpencil.grease_pencil_modifiers["Line Art"].source_type = 'OBJECT'
        gpencil.grease_pencil_modifiers["Line Art"].source_object = obj
        gpencil.grease_pencil_modifiers["Line Art"].target_layer = "Lines"
        gpencil.grease_pencil_modifiers["Line Art"].target_material = bpy.data.objects['GPencil'].material_slots[0].material
        gpencil.grease_pencil_modifiers["Line Art"].thickness = 10
        gpencil.show_in_front = True


def remove_materials(obj):
    set_active_object(obj)
    for _ in obj.material_slots.items():
        bpy.ops.object.material_slot_remove()


def add_material(obj, rgba=None, name=None):
    set_active_object(obj)

    bpy.ops.material.new()
    material = bpy.data.materials[-1]
    material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = rgba or (1, 1, 1, 1)
    material.shadow_method = 'NONE'
    material.node_tree.nodes["Principled BSDF"].inputs["Specular"].default_value = 0
    material.name = name or material.name
    return material


def setup_lights(cam, target_object):
    bpy.ops.object.light_add(type="SUN", location=[0, 0, cam.location[2]])
    overhead_light = bpy.context.object
    overhead_light.data.use_shadow = False
    bpy.ops.object.light_add(type="SUN", location=[cam.location[0], cam.location[1], 0])
    front_light = bpy.context.object
    front_light.parent = target_object
    front_light.rotation_euler[0] = math.radians(90)
    front_light.data.use_shadow = False
    front_light_constraint = front_light.constraints.new(type="TRACK_TO")
    front_light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    front_light_constraint.target = target_object

    bpy.ops.object.light_add(type="SUN", location=cam.location)
    cam_light = bpy.context.object
    cam_light.parent = target_object
    cam_light.data.use_shadow = False
    cam_light_constraint = cam_light.constraints.new(type="TRACK_TO")
    cam_light_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_light_constraint.target = target_object
    return overhead_light, front_light, cam_light


def get_image_path(input_file, output_dir, category):
    model_identifier = os.path.basename(os.path.dirname(os.path.dirname(input_file)))
    return os.path.join(os.path.abspath(output_dir), f"{os.path.basename(output_dir)}_{category}", model_identifier, model_identifier)


def save_images(image_path_prefix, cam_target, views=None):
    stepsize = 360.0 / args.views
    for i in range(0, views):
        print(f"Rotation {(stepsize * i)}, {math.radians(stepsize * i)}")
        render_file_path = f"{image_path_prefix}_{int(i * stepsize):03d}".format()
        bpy.context.scene.render.filepath = render_file_path
        bpy.ops.render.render(write_still=True)  # render still
        cam_target.rotation_euler[2] += math.radians(stepsize)


def set_active_object(obj=None):

    active_object = obj or next((o for o in bpy.context.scene.objects if 'model_normalized' in o.name.lower()))
    bpy.context.view_layer.objects.active = active_object
    return active_object


def get_image_paths(data_dir, split_file):
    print("getting image paths....")
    with open(split_file, 'r') as open_split_file:
        splits = json.load(open_split_file)
        total = 0
        for dataset, l1 in splits.items():
            if dataset.lower() == 'shapenetv2':
                total += sum(len(ids) for ids in l1.values())
                for l1_name, l2 in l1.items():
                    for idx, l2_name in enumerate(l2):
                        yield os.path.join(data_dir, dataset, l1_name, l2_name, "models", "model_normalized.obj")
            else:
                raise Exception("Split file dataset not supported yet.")

def delete_object(obj):
    set_active_object(obj).select_set(True)
    bpy.ops.object.delete(use_global=True, confirm=False)

def main(file_path, output_folder, view_count, import_options, data_dir=None):
    cam, cam_target = setup_camera()
    setup_lights(cam, cam_target)
    files = get_image_paths(data_dir, file_path) if data_dir else [file_path]
    print(files)
    for obj_file_path in files:
        print(f"obj_file_path: {obj_file_path}")

        obj = import_obj(obj_file_path, **import_options)
        add_line_art(obj)
        # save regular color "sketch" views
        save_images(get_image_path(obj_file_path, output_folder, "color"), cam_target, view_count)
        # save B/W "sketch" views
        remove_materials(obj)
        add_material(obj)
        # add_line_art(obj)
        save_images(get_image_path(obj_file_path, output_folder, "sketch"), cam_target, view_count)
        delete_object(obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Renders given obj file by rotation a camera around it.")
    parser.add_argument("--views", type=int, default=1, help="number of views to be rendered")
    parser.add_argument("obj", type=str, help="Path to the obj file to be rendered.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/tmp",
        help="The path the output will be dumped to.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path the output will be dumped to.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Scaling factor applied to model. Depends on size of mesh.",
    )
    parser.add_argument(
        "--remove_doubles",
        type=bool,
        default=True,
        help="Remove double vertices to improve mesh quality.",
    )
    parser.add_argument("--edge_split", type=bool, default=True, help="Adds edge split filter.")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1,
        help="Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.",
    )
    parser.add_argument(
        "--color_depth",
        type=str,
        default="8",
        help="Number of bit per channel used for output. Either 8 or 16.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="PNG",
        help="Format of files generated. Either PNG or OPEN_EXR",
    )
    parser.add_argument("--resolution", type=int, default=128, help="Resolution of the images.")
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        help="Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...",
    )

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    setup(args)
    main(args.obj, args.output_folder, args.views, import_options=args.__dict__, data_dir=args.data_dir)
