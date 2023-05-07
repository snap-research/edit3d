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
from glob import glob

parser = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
parser.add_argument(
    "--views", type=int, default=1, help="number of views to be rendered"
)
parser.add_argument("obj", type=str, help="Path to the obj file to be rendered.")
parser.add_argument(
    "--output_folder",
    type=str,
    default="/tmp",
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
parser.add_argument(
    "--edge_split", type=bool, default=True, help="Adds edge split filter."
)
parser.add_argument(
    "--depth_scale",
    type=float,
    default=1.4,
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
parser.add_argument(
    "--resolution", type=int, default=128, help="Resolution of the images."
)
parser.add_argument(
    "--engine",
    type=str,
    default="BLENDER_EEVEE",
    help="Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...",
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine
render.image_settings.color_mode = "RGBA"  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth  # ('8', '16')
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new("CompositorNodeRLayers")

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action="DESELECT")

# import pdb; pdb.set_trace()
# bpy.ops.import_scene.obj(filepath=args.obj)
bpy.ops.import_mesh.ply(filepath=args.obj)
# bpy.context.active_object.data.vertex_colors[0].data[0].color[2] # it loads color
# import pdb; pdb.set_trace()

obj = bpy.context.selected_objects[0]

mat = bpy.data.materials.new("material_1")
obj.active_material = mat
# mat.use_vertex_color_paint = True
mat.use_nodes = True
nodes = mat.node_tree.nodes
mat_links = mat.node_tree.links
bsdf = nodes.get("Principled BSDF")
assert bsdf  # make sure it exists to continue
vcol = nodes.new(type="ShaderNodeVertexColor")
# vcol.layer_name = "VColor" # the vertex color layer name
vcol.layer_name = "Col"
mat_links.new(vcol.outputs["Color"], bsdf.inputs["Base Color"])

from mathutils import Vector

minX = min([vertex.co[0] for vertex in obj.data.vertices])
minY = min([vertex.co[1] for vertex in obj.data.vertices])
minZ = min([vertex.co[2] for vertex in obj.data.vertices])
vMin = Vector([minX, minY, minZ])
maxX = max([vertex.co[0] for vertex in obj.data.vertices])
maxY = max([vertex.co[1] for vertex in obj.data.vertices])
maxZ = max([vertex.co[2] for vertex in obj.data.vertices])
vMax = Vector([maxX, maxY, maxZ])
center = (vMax + vMin) / 2

# import pdb; pdb.set_trace()
for v in obj.data.vertices:
    v.co -= center  # Set all coordinates start from (0, 0, 0)

max_dist = max(obj.dimensions)
for v in obj.data.vertices:
    v.co /= max_dist  # Set all coordinates between 0 and 1
    v.co *= 0.7  # make the object a bit smaller

# obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

for obj in bpy.context.selected_objects:
    context.view_layer.objects.active = obj
    context.object.cycles_visibility.shadow = False

# Possibly disable specular shading
for slot in obj.material_slots:
    node = slot.material.node_tree.nodes["Principled BSDF"]
    node.inputs["Specular"].default_value = 0

if args.scale != 1:
    bpy.ops.transform.resize(value=(args.scale, args.scale, args.scale))
    bpy.ops.object.transform_apply(scale=True)
if args.remove_doubles:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode="OBJECT")
if args.edge_split:
    bpy.ops.object.modifier_add(type="EDGE_SPLIT")
    context.object.modifiers["EdgeSplit"].split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Set objekt IDs
obj.pass_index = 1

# Make light just directional, disable shadows.
light = bpy.data.lights["Light"]
light.type = "SUN"
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 0.0
light.energy = 10

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type="SUN")
light2 = bpy.data.lights["Sun"]
light2.use_shadow = False
light2.specular_factor = 0.0
light2.energy = 0.015
bpy.data.objects["Sun"].rotation_euler = bpy.data.objects["Light"].rotation_euler
bpy.data.objects["Sun"].rotation_euler[0] += 180

# Place camera
cam = scene.objects["Camera"]
cam.location = (0, 1, 0.6)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = "XYZ"

model_identifier = os.path.split(os.path.split(os.path.split(args.obj)[0])[0])[1]
fp = os.path.join(os.path.abspath(args.output_folder), model_identifier, "render")

# bpy.ops.object.mode_set(mode='VERTEX_PAINT')

for i in range(0, args.views):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

    render_file_path = fp + "_r_{0:03d}".format(int(i * stepsize))

    scene.render.filepath = render_file_path

    bpy.ops.render.render(write_still=True)  # render still

    cam_empty.rotation_euler[2] += math.radians(stepsize)

# For debugging the workflow
# bpy.ops.wm.save_as_mainfile(filepath='debug.blend')
